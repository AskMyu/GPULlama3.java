package org.beehive.gpullama3.isolation.serialization;

import org.beehive.gpullama3.isolation.core.ProcessSerializer;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Optimized serializer for TornadoVM FloatArray objects.
 * Uses efficient binary format with optional compression for large arrays.
 */
public class FloatArraySerializer implements ProcessSerializer<FloatArray> {
    private final boolean compressionEnabled;
    private static final int COMPRESSION_THRESHOLD = 1024 * 1024; // 1MB
    private static final byte[] MAGIC_BYTES = {'T', 'F', 'A', '1'}; // TornadoFloatArray v1
    
    public FloatArraySerializer() {
        this(false);
    }
    
    public FloatArraySerializer(boolean compressionEnabled) {
        this.compressionEnabled = compressionEnabled;
    }
    
    @Override
    public void serialize(FloatArray floatArray, OutputStream output) throws SerializationException {
        try {
            System.err.printf("[SERIALIZER] Starting FloatArray serialization: %d elements%n", floatArray.getSize());
            
            DataOutputStream dos = new DataOutputStream(output);
            
            // Write magic bytes
            dos.write(MAGIC_BYTES);
            System.err.printf("[SERIALIZER] Magic bytes written%n");
            
            int size = floatArray.getSize();
            boolean useCompression = compressionEnabled && (size * 4) > COMPRESSION_THRESHOLD;
            
            // Write header
            dos.writeInt(size);
            dos.writeBoolean(useCompression);
            System.err.printf("[SERIALIZER] Header written: size=%d, compression=%s%n", size, useCompression);
            
            // Serialize data
            if (useCompression) {
                System.err.printf("[SERIALIZER] Using compression for large array%n");
                serializeCompressed(floatArray, dos);
            } else {
                System.err.printf("[SERIALIZER] Using uncompressed serialization%n");
                serializeUncompressed(floatArray, dos);
            }
            
            dos.flush();
            System.err.printf("[SERIALIZER] Serialization completed successfully%n");
            
        } catch (IOException e) {
            System.err.printf("[SERIALIZER] IOException during serialization: %s%n", e.getMessage());
            e.printStackTrace();
            throw new SerializationException("Failed to serialize FloatArray: " + e.getMessage(), e);
        } catch (Exception e) {
            System.err.printf("[SERIALIZER] Unexpected error during serialization: %s%n", e.getMessage());
            e.printStackTrace();
            throw new SerializationException("Unexpected error during serialization: " + e.getMessage(), e);
        }
    }
    
    @Override
    public FloatArray deserialize(InputStream input) throws SerializationException {
        try (DataInputStream dis = new DataInputStream(input)) {
            // Check if input is available
            if (input.available() == 0) {
                throw new SerializationException("No input data available for deserialization");
            }
            
            // Verify magic bytes
            byte[] magic = new byte[4];
            try {
                dis.readFully(magic);
            } catch (java.io.EOFException e) {
                throw new SerializationException("Incomplete input - missing magic bytes", e);
            }
            
            if (!java.util.Arrays.equals(magic, MAGIC_BYTES)) {
                throw new SerializationException("Invalid FloatArray format - magic bytes mismatch");
            }
            
            // Read header
            int size = dis.readInt();
            boolean isCompressed = dis.readBoolean();
            
            if (size <= 0 || size > 100_000_000) { // Sanity check
                throw new SerializationException("Invalid FloatArray size: " + size);
            }
            
            // Deserialize data
            FloatArray result = new FloatArray(size);
            
            if (isCompressed) {
                deserializeCompressed(dis, result, size);
            } else {
                deserializeUncompressed(dis, result, size);
            }
            
            return result;
            
        } catch (IOException e) {
            throw new SerializationException("Failed to deserialize FloatArray", e);
        }
    }
    
    private void serializeUncompressed(FloatArray floatArray, DataOutputStream dos) throws IOException {
        int size = floatArray.getSize();
        System.err.printf("[SERIALIZER] Serializing uncompressed: %d elements%n", size);
        
        // Use ByteBuffer for efficient bulk operations
        int bufferSize = Math.min(8192, size * 4);
        ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        int processed = 0;
        int batchCount = 0;
        while (processed < size) {
            buffer.clear();
            
            int batchSize = Math.min((buffer.capacity() / 4), size - processed);
            
            try {
                for (int i = 0; i < batchSize; i++) {
                    float value = floatArray.get(processed + i);
                    buffer.putFloat(value);
                }
                
                dos.write(buffer.array(), 0, batchSize * 4);
                processed += batchSize;
                batchCount++;
                
                // Log progress every 1000 batches or at end
                if (batchCount % 1000 == 0 || processed >= size) {
                    System.err.printf("[SERIALIZER] Progress: %d/%d elements (batch %d)%n", 
                        processed, size, batchCount);
                }
                
            } catch (Exception e) {
                System.err.printf("[SERIALIZER] Error in batch %d at element %d: %s%n", 
                    batchCount, processed, e.getMessage());
                throw e;
            }
        }
        
        System.err.printf("[SERIALIZER] Uncompressed serialization completed: %d batches%n", batchCount);
    }
    
    private void serializeCompressed(FloatArray floatArray, DataOutputStream dos) throws IOException {
        try (GZIPOutputStream gzipOut = new GZIPOutputStream(dos)) {
            DataOutputStream compressedDos = new DataOutputStream(gzipOut);
            serializeUncompressed(floatArray, compressedDos);
            compressedDos.flush();
        }
    }
    
    private void deserializeUncompressed(DataInputStream dis, FloatArray result, int size) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(Math.min(8192, size * 4));
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        int processed = 0;
        while (processed < size) {
            int batchSize = Math.min((buffer.capacity() / 4), size - processed);
            int bytesToRead = batchSize * 4;
            
            buffer.clear();
            dis.readFully(buffer.array(), 0, bytesToRead);
            buffer.limit(bytesToRead);
            
            for (int i = 0; i < batchSize; i++) {
                result.set(processed + i, buffer.getFloat());
            }
            
            processed += batchSize;
        }
    }
    
    private void deserializeCompressed(DataInputStream dis, FloatArray result, int size) throws IOException {
        try (GZIPInputStream gzipIn = new GZIPInputStream(dis)) {
            DataInputStream compressedDis = new DataInputStream(gzipIn);
            deserializeUncompressed(compressedDis, result, size);
        }
    }
    
    @Override
    public String getSerializationFormat() {
        return compressionEnabled ? "BINARY_COMPRESSED" : "BINARY";
    }
    
    /**
     * Estimate serialized size for memory planning.
     */
    public long estimateSerializedSize(FloatArray floatArray) {
        long baseSize = 4 + 4 + 1 + (floatArray.getSize() * 4L); // magic + size + compression flag + data
        
        if (compressionEnabled && (floatArray.getSize() * 4) > COMPRESSION_THRESHOLD) {
            // Estimate ~30-50% compression ratio for float data
            return baseSize / 2;
        }
        
        return baseSize;
    }
}