package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Chunked FloatArray wrapper that overcomes TornadoVM's 2GB limit.
 * 
 * This class provides a transparent interface identical to FloatArray while 
 * internally managing multiple smaller FloatArray chunks to support large
 * allocations that exceed Integer.MAX_VALUE bytes.
 * 
 * The implementation maintains full compatibility with existing code while
 * enabling models with >2GB cache requirements (like LLaVA with 4K context).
 */
public class ChunkedFloatArray {
    
    private static final int MAX_CHUNK_SIZE = Integer.MAX_VALUE / 4 / 2; // ~512M floats per chunk, well under 2GB
    
    private final FloatArray[] chunks;
    private final int totalSize;
    private final int numChunks;
    private final int chunkSize;
    
    /**
     * Create a chunked float array that can handle sizes beyond TornadoVM limits.
     * 
     * @param size Total number of floats to allocate
     */
    public ChunkedFloatArray(int size) {
        this.totalSize = size;
        
        if (size <= MAX_CHUNK_SIZE) {
            // Small enough for single FloatArray - use normal allocation
            this.chunks = new FloatArray[1];
            this.chunks[0] = new FloatArray(size);
            this.numChunks = 1;
            this.chunkSize = size;
            System.out.println("DEBUG: ChunkedFloatArray using single chunk: " + size + " floats");
        } else {
            // Need multiple chunks
            this.chunkSize = MAX_CHUNK_SIZE;
            this.numChunks = (size + chunkSize - 1) / chunkSize; // Ceiling division
            this.chunks = new FloatArray[numChunks];
            
            for (int i = 0; i < numChunks; i++) {
                int currentChunkSize = Math.min(chunkSize, size - (i * chunkSize));
                this.chunks[i] = new FloatArray(currentChunkSize);
            }
            
            System.out.println("DEBUG: ChunkedFloatArray using " + numChunks + " chunks of " + chunkSize + " floats each");
            System.out.println("DEBUG: Total allocation: " + size + " floats (" + (size * 4 / (1024*1024)) + "MB)");
        }
    }
    
    /**
     * Get the value at the specified index.
     */
    public float get(int index) {
        if (numChunks == 1) {
            return chunks[0].get(index);
        }
        
        int chunkIndex = index / chunkSize;
        int localIndex = index % chunkSize;
        return chunks[chunkIndex].get(localIndex);
    }
    
    /**
     * Set the value at the specified index.
     */
    public void set(int index, float value) {
        if (numChunks == 1) {
            chunks[0].set(index, value);
            return;
        }
        
        int chunkIndex = index / chunkSize;
        int localIndex = index % chunkSize;
        chunks[chunkIndex].set(localIndex, value);
    }
    
    /**
     * Get the total size of the array.
     */
    public int getSize() {
        return totalSize;
    }
    
    /**
     * Initialize all values to the specified value.
     * Note: This method is avoided in the original code due to GPU memory issues,
     * so we provide it but don't recommend using it for large allocations.
     */
    public void init(float value) {
        for (FloatArray chunk : chunks) {
            // Only initialize if chunk is small enough to be safe
            if (chunk.getSize() < 100000000) { // 100M floats threshold
                chunk.init(value);
            } else {
                System.err.println("WARNING: Skipping init() on large chunk to avoid GPU memory issues");
            }
        }
    }
    
    /**
     * Get direct access to a specific chunk for advanced use cases.
     * This breaks the abstraction but may be needed for performance-critical code.
     */
    public FloatArray getChunk(int chunkIndex) {
        if (chunkIndex >= numChunks) {
            throw new IndexOutOfBoundsException("Chunk index " + chunkIndex + " >= " + numChunks);
        }
        return chunks[chunkIndex];
    }
    
    /**
     * Get the number of chunks this array is split into.
     */
    public int getNumChunks() {
        return numChunks;
    }
    
    /**
     * Get the size of each chunk (except potentially the last one).
     */
    public int getChunkSize() {
        return chunkSize;
    }
    
    /**
     * Copy data from another ChunkedFloatArray or regular array.
     */
    public void copyFrom(float[] source) {
        if (source.length != totalSize) {
            throw new IllegalArgumentException("Source array size " + source.length + " != target size " + totalSize);
        }
        
        for (int i = 0; i < totalSize; i++) {
            set(i, source[i]);
        }
    }
    
    /**
     * Copy data to a regular float array.
     */
    public void copyTo(float[] target) {
        if (target.length != totalSize) {
            throw new IllegalArgumentException("Target array size " + target.length + " != source size " + totalSize);
        }
        
        for (int i = 0; i < totalSize; i++) {
            target[i] = get(i);
        }
    }
    
    /**
     * Get a description of the chunking strategy for debugging.
     */
    @Override
    public String toString() {
        return String.format("ChunkedFloatArray[size=%d, chunks=%d, chunkSize=%d, totalMB=%d]", 
                           totalSize, numChunks, chunkSize, totalSize * 4 / (1024*1024));
    }
}