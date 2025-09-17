package org.beehive.gpullama3.isolation.serialization;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import org.beehive.gpullama3.isolation.core.ProcessSerializer;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * Memory-mapped file based serializer for reliable subprocess communication.
 * Eliminates pipe buffer issues and provides persistent debugging capability.
 */
public class MemoryMappedFileSerializer implements ProcessSerializer<FloatArray> {
    
    private static final int MAGIC_BYTES = 0x464C4F41; // "FLOA" in ASCII
    private static final int HEADER_SIZE = 16; // magic(4) + size(4) + checksum(4) + reserved(4)
    
    private final String processId;
    private final Path tempDir;
    
    public MemoryMappedFileSerializer(String processId) {
        this.processId = processId;
        this.tempDir = Paths.get("/tmp");
    }
    
    /**
     * Serialize FloatArray to memory-mapped file.
     * For parent process: writes to input file and sends path to subprocess via stdout
     * For subprocess: writes to output file and sends path back to parent via stdout
     */
    @Override
    public void serialize(FloatArray data, OutputStream outputStream) throws SerializationException {
        try {
            // Determine if this is parent->subprocess (input) or subprocess->parent (output)
            boolean isSubprocess = System.getProperty("gpu.process.isolation") != null;
            Path targetFile = isSubprocess ? getOutputFilePath() : getInputFilePath();
            
            System.err.printf("[MMAP-SERIALIZER] Writing %d elements to: %s (isSubprocess=%s)%n", 
                data.getSize(), targetFile, isSubprocess);
            
            int dataSize = data.getSize();
            long fileSize = HEADER_SIZE + (dataSize * 4L); // 4 bytes per float
            
            // Create and write to memory-mapped file
            try (FileChannel channel = FileChannel.open(targetFile, 
                    StandardOpenOption.CREATE, 
                    StandardOpenOption.READ,
                    StandardOpenOption.WRITE, 
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                
                MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_WRITE, 0, fileSize);
                buffer.order(ByteOrder.LITTLE_ENDIAN);
                
                // Write header
                buffer.putInt(MAGIC_BYTES);
                buffer.putInt(dataSize);
                buffer.putInt(0); // checksum placeholder
                buffer.putInt(0); // reserved
                
                // Write data and calculate checksum
                int checksum = 0;
                for (int i = 0; i < dataSize; i++) {
                    float value = data.get(i);
                    buffer.putFloat(value);
                    checksum ^= Float.floatToIntBits(value);
                }
                
                // Update checksum in header
                buffer.putInt(8, checksum);
                buffer.force();
                
                System.err.printf("[MMAP-SERIALIZER] Successfully wrote %d elements with checksum 0x%08X%n", 
                    dataSize, checksum);
            }
            
            // Send file path to counterpart process
            try (PrintWriter writer = new PrintWriter(outputStream)) {
                writer.println(targetFile.toString());
                writer.flush();
            }
            
        } catch (Exception e) {
            throw new SerializationException("Failed to serialize to memory-mapped file", e);
        }
    }
    
    /**
     * Deserialize FloatArray from memory-mapped file.
     */
    @Override
    public FloatArray deserialize(InputStream inputStream) throws SerializationException {
        try {
            // Read file path from inputStream
            String filePath;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
                filePath = reader.readLine();
                if (filePath == null || filePath.trim().isEmpty()) {
                    throw new SerializationException("No file path received from subprocess");
                }
            }
            
            Path outputFile = Paths.get(filePath);
            System.err.printf("[MMAP-SERIALIZER] Reading from: %s%n", outputFile);
            
            if (!Files.exists(outputFile)) {
                throw new SerializationException("Output file does not exist: " + outputFile);
            }
            
            // Read from memory-mapped file
            try (FileChannel channel = FileChannel.open(outputFile, StandardOpenOption.READ)) {
                long fileSize = channel.size();
                if (fileSize < HEADER_SIZE) {
                    throw new SerializationException("File too small: " + fileSize + " bytes");
                }
                
                MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
                buffer.order(ByteOrder.LITTLE_ENDIAN);
                
                // Read and validate header
                int magic = buffer.getInt();
                if (magic != MAGIC_BYTES) {
                    throw new SerializationException(String.format("Invalid magic bytes: 0x%08X", magic));
                }
                
                int dataSize = buffer.getInt();
                int storedChecksum = buffer.getInt();
                buffer.getInt(); // reserved
                
                if (dataSize <= 0 || dataSize > 10_000_000) {
                    throw new SerializationException("Invalid data size: " + dataSize);
                }
                
                long expectedFileSize = HEADER_SIZE + (dataSize * 4L);
                if (fileSize != expectedFileSize) {
                    throw new SerializationException(String.format("File size mismatch: %d vs expected %d", 
                        fileSize, expectedFileSize));
                }
                
                // Read data and verify checksum
                FloatArray result = new FloatArray(dataSize);
                int checksum = 0;
                
                for (int i = 0; i < dataSize; i++) {
                    float value = buffer.getFloat();
                    result.set(i, value);
                    checksum ^= Float.floatToIntBits(value);
                }
                
                if (checksum != storedChecksum) {
                    throw new SerializationException(String.format("Checksum mismatch: 0x%08X vs 0x%08X", 
                        checksum, storedChecksum));
                }
                
                System.err.printf("[MMAP-SERIALIZER] Successfully read %d elements with verified checksum%n", 
                    dataSize);
                
                return result;
            }
            
        } catch (Exception e) {
            throw new SerializationException("Failed to deserialize from memory-mapped file", e);
        }
    }
    
    /**
     * Get input file path for subprocess communication.
     */
    public Path getInputFilePath() {
        return tempDir.resolve("mlp_input_" + processId + ".bin");
    }
    
    /**
     * Get output file path for subprocess communication.
     */
    public Path getOutputFilePath() {
        return tempDir.resolve("mlp_output_" + processId + ".bin");
    }
    
    /**
     * Clean up temporary files.
     */
    public void cleanup() {
        try {
            Files.deleteIfExists(getInputFilePath());
            Files.deleteIfExists(getOutputFilePath());
            System.err.printf("[MMAP-SERIALIZER] Cleaned up temp files for process %s%n", processId);
        } catch (Exception e) {
            System.err.printf("[MMAP-SERIALIZER] Warning: Failed to cleanup temp files: %s%n", e.getMessage());
        }
    }
    
    @Override
    public String getSerializationFormat() {
        return "memory-mapped-file";
    }
}