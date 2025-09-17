/*
 * This file is part of GPULlama3.java (https://github.com/beehive-lab/GPULlama3.java)
 *
 * GPULlama3.java is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GPULlama3.java is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GPULlama3.java. If not, see <http://www.gnu.org/licenses/>.
 */
package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Vision Prefill Batch State Container for TornadoVM Parallel GPU Processing
 * 
 * This class enables parallel processing of multiple vision positions simultaneously
 * using TornadoVM's parallel execution model. Instead of processing 144 vision positions
 * sequentially, this allows batches of 4-8 positions to be processed in parallel.
 * 
 * Memory Layout:
 * - qBatch[batchSize]: Query vectors for each position in the batch
 * - keyCacheBatch[batchSize][numLayers]: Key cache arrays per position per layer
 * - valueCacheBatch[batchSize][numLayers]: Value cache arrays per position per layer
 * - outputBatch[batchSize]: Output arrays for each position
 * 
 * Expected Performance Improvement: 4-6x speedup in vision prefill phase
 * 
 * Used by: visionPrefillAttentionKernelGPUBatch and batch processing pipelines
 */
public class VisionPrefillBatchState {
    
    // ========== BATCH DIMENSIONS ==========
    /** Number of positions processed simultaneously in this batch */
    public int batchSize;
    
    /** Number of transformer layers (typically 32) */
    public int numLayers;
    
    // ========== BATCH ATTENTION ARRAYS ==========
    /** Input vision embeddings for each position: [batchSize][dim] */
    public FloatArray[] inputBatch;
    
    /** Query vectors for each position: [batchSize][seqLen * dim] */
    public FloatArray[] qBatch;
    
    /** Key cache per position per layer: [batchSize][numLayers][heads * seqLen * headSize] */
    public FloatArray[][] keyCacheBatch;
    
    /** Value cache per position per layer: [batchSize][numLayers][heads * seqLen * headSize] */
    public FloatArray[][] valueCacheBatch;
    
    /** Output array per position: [batchSize][seqLen * dim] */
    public FloatArray[] outputBatch;
    
    // ========== CACHED CONCATENATED ARRAYS ==========
    /** Cached concatenated key arrays per layer to prevent recreation */
    private FloatArray[] cachedBatchKeyCache;
    /** Cached concatenated value arrays per layer to prevent recreation */
    private FloatArray[] cachedBatchValueCache;
    
    // ========== MODEL DIMENSIONS (SHARED ACROSS BATCH) ==========
    /** Number of attention heads */
    public int nHeads;
    
    /** Size of each attention head (dim / nHeads) */
    public int headSize;
    
    /** Key-Value dimension (for GQA - may differ from nHeads) */
    public int kvDim;
    
    /** Key-Value multiplier for grouped query attention */
    public int kvMul;
    
    // ========== BATCH POSITION INFORMATION ==========
    /** Starting position for this batch (e.g., 0, 8, 16, ...) */
    public int batchStartPosition;
    
    /** Maximum context length */
    public int contextLength;
    
    /** Sequence length for vision tokens */
    public int seqLen;
    
    /** Model dimension */
    public int dim;
    
    // ========== CONSTRUCTORS ==========
    
    /**
     * Default constructor for TornadoVM compatibility
     */
    public VisionPrefillBatchState() {
        // Empty constructor required for TornadoVM object serialization
    }
    
    /**
     * Full constructor for batch vision prefill state initialization
     * 
     * @param batchSize Number of positions to process simultaneously
     * @param numLayers Number of transformer layers
     * @param seqLen Sequence length for vision tokens
     * @param dim Model dimension
     * @param nHeads Number of attention heads
     * @param headSize Size of each head
     * @param kvDim Key-value dimension
     * @param kvMul Key-value multiplier
     * @param batchStartPosition Starting position for this batch
     * @param contextLength Maximum context length
     */
    public VisionPrefillBatchState(int batchSize, int numLayers, int seqLen, int dim, 
                                  int nHeads, int headSize, int kvDim, int kvMul, 
                                  int batchStartPosition, int contextLength) {
        this.batchSize = batchSize;
        this.numLayers = numLayers;
        this.seqLen = seqLen;
        this.dim = dim;
        this.nHeads = nHeads;
        this.headSize = headSize;
        this.kvDim = kvDim;
        this.kvMul = kvMul;
        this.batchStartPosition = batchStartPosition;
        this.contextLength = contextLength;
        
        // Pre-allocate all GPU memory to avoid CL_OUT_OF_RESOURCES during execution
        initializeBatchArrays();
    }
    
    // ========== MEMORY MANAGEMENT ==========
    
    /**
     * Pre-allocates all GPU memory arrays for the batch
     * Critical for avoiding CL_OUT_OF_RESOURCES errors during TornadoVM execution
     */
    private void initializeBatchArrays() {
        // Allocate input batch arrays (vision embeddings)
        inputBatch = new FloatArray[batchSize];
        for (int i = 0; i < batchSize; i++) {
            inputBatch[i] = new FloatArray(dim);
        }
        
        // Allocate query batch arrays
        qBatch = new FloatArray[batchSize];
        for (int i = 0; i < batchSize; i++) {
            qBatch[i] = new FloatArray(seqLen * dim);
        }
        
        // MEMORY OPTIMIZATION: Allocate smaller cache arrays per position per layer
        // Instead of pre-allocating for all layers, allocate only what's needed for current layer
        keyCacheBatch = new FloatArray[batchSize][numLayers];
        valueCacheBatch = new FloatArray[batchSize][numLayers];
        
        // Initialize arrays to null - allocate on demand in the kernel
        for (int pos = 0; pos < batchSize; pos++) {
            for (int layer = 0; layer < numLayers; layer++) {
                keyCacheBatch[pos][layer] = null;  // Allocate on-demand
                valueCacheBatch[pos][layer] = null; // Allocate on-demand
            }
        }
        
        // Allocate output batch arrays
        outputBatch = new FloatArray[batchSize];
        for (int i = 0; i < batchSize; i++) {
            outputBatch[i] = new FloatArray(seqLen * dim);
        }
        
        // Initialize cached concatenated arrays
        cachedBatchKeyCache = new FloatArray[numLayers];
        cachedBatchValueCache = new FloatArray[numLayers];
        // Arrays are initialized to null - created on first access
    }
    
    // ========== ACCESSOR METHODS ==========
    
    /**
     * Get query array for specific position in batch
     * @param positionInBatch Position index within batch (0 to batchSize-1)
     * @return Query FloatArray for the position
     */
    public FloatArray getQForPosition(int positionInBatch) {
        if (positionInBatch < 0 || positionInBatch >= batchSize) {
            throw new IllegalArgumentException("Position " + positionInBatch + " out of batch range [0, " + (batchSize-1) + "]");
        }
        return qBatch[positionInBatch];
    }
    
    /**
     * Get key cache array for specific position and layer
     * @param positionInBatch Position index within batch (0 to batchSize-1)
     * @param layer Layer index (0 to numLayers-1)
     * @return Key cache FloatArray for the position and layer
     */
    public FloatArray getKeyCacheForPosition(int positionInBatch, int layer) {
        if (positionInBatch < 0 || positionInBatch >= batchSize) {
            throw new IllegalArgumentException("Position " + positionInBatch + " out of batch range [0, " + (batchSize-1) + "]");
        }
        if (layer < 0 || layer >= numLayers) {
            throw new IllegalArgumentException("Layer " + layer + " out of range [0, " + (numLayers-1) + "]");
        }
        return keyCacheBatch[positionInBatch][layer];
    }
    
    /**
     * Get value cache array for specific position and layer
     * @param positionInBatch Position index within batch (0 to batchSize-1)
     * @param layer Layer index (0 to numLayers-1)
     * @return Value cache FloatArray for the position and layer
     */
    public FloatArray getValueCacheForPosition(int positionInBatch, int layer) {
        if (positionInBatch < 0 || positionInBatch >= batchSize) {
            throw new IllegalArgumentException("Position " + positionInBatch + " out of batch range [0, " + (batchSize-1) + "]");
        }
        if (layer < 0 || layer >= numLayers) {
            throw new IllegalArgumentException("Layer " + layer + " out of range [0, " + (numLayers-1) + "]");
        }
        return valueCacheBatch[positionInBatch][layer];
    }
    
    /**
     * Get output array for specific position in batch
     * @param positionInBatch Position index within batch (0 to batchSize-1)
     * @return Output FloatArray for the position
     */
    public FloatArray getOutputForPosition(int positionInBatch) {
        if (positionInBatch < 0 || positionInBatch >= batchSize) {
            throw new IllegalArgumentException("Position " + positionInBatch + " out of batch range [0, " + (batchSize-1) + "]");
        }
        return outputBatch[positionInBatch];
    }
    
    /**
     * Get the actual global position for a position within this batch
     * @param positionInBatch Position index within batch (0 to batchSize-1)
     * @return Global position (e.g., if batch starts at 8 and positionInBatch=2, returns 10)
     */
    public int getGlobalPosition(int positionInBatch) {
        return batchStartPosition + positionInBatch;
    }
    
    // ========== UTILITY METHODS ==========
    
    /**
     * Validates that all required fields are properly initialized
     * @return true if batch state is valid for GPU processing
     */
    public boolean isValid() {
        if (batchSize <= 0 || numLayers <= 0 || seqLen <= 0 || dim <= 0) return false;
        if (nHeads <= 0 || headSize <= 0 || kvDim <= 0 || kvMul <= 0) return false;
        if (batchStartPosition < 0 || contextLength <= 0) return false;
        
        // Check array allocations
        if (qBatch == null || qBatch.length != batchSize) return false;
        if (keyCacheBatch == null || keyCacheBatch.length != batchSize) return false;
        if (valueCacheBatch == null || valueCacheBatch.length != batchSize) return false;
        if (outputBatch == null || outputBatch.length != batchSize) return false;
        
        // Check individual array allocations (allowing on-demand allocation for cache arrays)
        for (int i = 0; i < batchSize; i++) {
            if (qBatch[i] == null || outputBatch[i] == null) return false;
            if (keyCacheBatch[i] == null || keyCacheBatch[i].length != numLayers) return false;
            if (valueCacheBatch[i] == null || valueCacheBatch[i].length != numLayers) return false;
            
            // MEMORY OPTIMIZATION: Allow null cache arrays (allocated on-demand)
            // This is valid for memory-optimized batch processing
        }
        
        return true;
    }
    
    /**
     * Calculates estimated GPU memory usage for this batch
     * @return Estimated memory usage in bytes
     */
    public long estimateMemoryUsage() {
        long floatSize = 4; // 4 bytes per float
        
        // Query arrays: batchSize * seqLen * dim * 4 bytes
        long qMemory = (long) batchSize * seqLen * dim * floatSize;
        
        // Key/Value cache: batchSize * numLayers * nHeads * seqLen * headSize * 4 bytes * 2 (key + value)
        long cacheMemory = (long) batchSize * numLayers * nHeads * seqLen * headSize * floatSize * 2;
        
        // Output arrays: batchSize * seqLen * dim * 4 bytes  
        long outputMemory = (long) batchSize * seqLen * dim * floatSize;
        
        return qMemory + cacheMemory + outputMemory;
    }
    
    /**
     * Returns a string representation for debugging
     */
    @Override
    public String toString() {
        long memoryMB = estimateMemoryUsage() / (1024 * 1024);
        return String.format(
            "VisionPrefillBatchState{batchSize=%d, layers=%d, startPos=%d, " +
            "nHeads=%d, headSize=%d, kvDim=%d, kvMul=%d, seqLen=%d, dim=%d, " +
            "memoryUsage=%dMB, arrays=%s}",
            batchSize, numLayers, batchStartPosition, nHeads, headSize, kvDim, kvMul,
            seqLen, dim, memoryMB, isValid() ? "OK" : "INVALID"
        );
    }
    
    // ========== TORNADOVM BATCH ARRAY METHODS ==========
    
    /**
     * Creates concatenated Q arrays for TornadoVM batch processing.
     * @return FloatArray containing all Q vectors concatenated
     */
    public FloatArray getBatchQ() {
        int totalSize = batchSize * nHeads * headSize;
        FloatArray batchQ = new FloatArray(totalSize);
        
        for (int pos = 0; pos < batchSize; pos++) {
            FloatArray posQ = qBatch[pos];
            int destOffset = pos * nHeads * headSize;
            
            for (int i = 0; i < nHeads * headSize; i++) {
                batchQ.set(destOffset + i, posQ.get(i));
            }
        }
        
        return batchQ;
    }
    
    /**
     * Creates concatenated input arrays for TornadoVM batch processing (vision embeddings).
     * @return FloatArray containing all vision embeddings concatenated
     */
    public FloatArray getBatchInput() {
        int totalSize = batchSize * dim;
        FloatArray batchInput = new FloatArray(totalSize);
        
        for (int pos = 0; pos < batchSize; pos++) {
            FloatArray posInput = inputBatch[pos];
            int destOffset = pos * dim;
            
            for (int i = 0; i < dim; i++) {
                batchInput.set(destOffset + i, posInput.get(i));
            }
        }
        
        return batchInput;
    }
    
    /**
     * FIXED: Returns CACHED concatenated key cache arrays for TornadoVM batch processing.
     * This ensures the same array is returned consistently across multiple calls.
     * MEMORY OPTIMIZED: Only allocates for single layer to reduce memory usage
     * @param layer Current layer being processed
     * @return Cached FloatArray containing key caches for current layer only
     */
    public FloatArray getBatchKeyCache(int layer) {
        // Return cached array if already created
        if (cachedBatchKeyCache[layer] != null) {
            return cachedBatchKeyCache[layer];
        }
        
        // Create and cache the concatenated array on first access
        int singleLayerSize = batchSize * nHeads * headSize;
        FloatArray batchKeyCache = new FloatArray(singleLayerSize);
        
        for (int pos = 0; pos < batchSize; pos++) {
            // Allocate on-demand if not already allocated
            if (keyCacheBatch[pos][layer] == null) {
                keyCacheBatch[pos][layer] = new FloatArray(nHeads * headSize);
            }
            
            FloatArray posKeyCache = keyCacheBatch[pos][layer];
            int destOffset = pos * nHeads * headSize;
            
            for (int i = 0; i < nHeads * headSize; i++) {
                batchKeyCache.set(destOffset + i, posKeyCache.get(i));
            }
        }
        
        // Cache and return
        cachedBatchKeyCache[layer] = batchKeyCache;
        return batchKeyCache;
    }
    
    /**
     * FIXED: Returns CACHED concatenated value cache arrays for TornadoVM batch processing.
     * This ensures the same array is returned consistently across multiple calls.
     * MEMORY OPTIMIZED: Only allocates for single layer to reduce memory usage
     * @param layer Current layer being processed
     * @return Cached FloatArray containing value caches for current layer only
     */
    public FloatArray getBatchValueCache(int layer) {
        // Return cached array if already created
        if (cachedBatchValueCache[layer] != null) {
            return cachedBatchValueCache[layer];
        }
        
        // Create and cache the concatenated array on first access
        int singleLayerSize = batchSize * nHeads * headSize;
        FloatArray batchValueCache = new FloatArray(singleLayerSize);
        
        for (int pos = 0; pos < batchSize; pos++) {
            // Allocate on-demand if not already allocated
            if (valueCacheBatch[pos][layer] == null) {
                valueCacheBatch[pos][layer] = new FloatArray(nHeads * headSize);
            }
            
            FloatArray posValueCache = valueCacheBatch[pos][layer];
            int destOffset = pos * nHeads * headSize;
            
            for (int i = 0; i < nHeads * headSize; i++) {
                batchValueCache.set(destOffset + i, posValueCache.get(i));
            }
        }
        
        // Cache and return
        cachedBatchValueCache[layer] = batchValueCache;
        return batchValueCache;
    }
    
    /**
     * Creates concatenated output arrays for TornadoVM batch processing.
     * @return FloatArray containing all output arrays concatenated
     */
    public FloatArray getBatchOutput() {
        int totalSize = batchSize * nHeads * headSize;
        FloatArray batchOutput = new FloatArray(totalSize);
        
        // Initialize to zero
        for (int i = 0; i < totalSize; i++) {
            batchOutput.set(i, 0.0f);
        }
        
        return batchOutput;
    }
    
    /**
     * Creates position array for TornadoVM batch processing.
     * @return IntArray containing positions for each batch element
     */
    public uk.ac.manchester.tornado.api.types.arrays.IntArray getBatchPositions() {
        uk.ac.manchester.tornado.api.types.arrays.IntArray batchPositions = 
            new uk.ac.manchester.tornado.api.types.arrays.IntArray(batchSize);
            
        for (int pos = 0; pos < batchSize; pos++) {
            batchPositions.set(pos, batchStartPosition + pos);
        }
        
        return batchPositions;
    }
}