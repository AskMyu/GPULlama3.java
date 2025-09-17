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
 * Vision Prefill State Container for TornadoVM GPU Acceleration
 * 
 * This class encapsulates all parameters required for vision prefill attention kernels,
 * solving TornadoVM's Task15 parameter limitation by using a single state object instead
 * of 11+ individual parameters.
 * 
 * Used by: visionPrefillAttentionKernel and related GPU-accelerated vision processing
 */
public class VisionPrefillState {
    
    // ========== ATTENTION ARRAYS ==========
    /** Query vectors for current position */
    public FloatArray q;
    
    /** Key cache (KV cache) for attention computation */
    public FloatArray keyCache;
    
    /** Value cache (KV cache) for attention computation */
    public FloatArray valueCache;
    
    /** Output array for attention results */
    public FloatArray output;
    
    // ========== MODEL DIMENSIONS ==========
    /** Number of attention heads */
    public int nHeads;
    
    /** Size of each attention head (dim / nHeads) */
    public int headSize;
    
    /** Key-Value dimension (for GQA - may differ from nHeads) */
    public int kvDim;
    
    /** Key-Value multiplier for grouped query attention */
    public int kvMul;
    
    // ========== POSITION INFORMATION ==========
    /** Current sequence position (for vision tokens: 0 to 143) */
    public int position;
    
    /** Maximum context length */
    public int contextLength;
    
    // ========== CONSTRUCTORS ==========
    
    /**
     * Default constructor for TornadoVM compatibility
     */
    public VisionPrefillState() {
        // Empty constructor required for TornadoVM object serialization
    }
    
    /**
     * Full constructor for vision prefill state initialization
     * 
     * @param q Query vectors
     * @param keyCache Key cache array
     * @param valueCache Value cache array  
     * @param output Output array for results
     * @param nHeads Number of attention heads
     * @param headSize Size of each head
     * @param kvDim Key-value dimension
     * @param kvMul Key-value multiplier
     * @param position Current sequence position
     * @param contextLength Maximum context length
     */
    public VisionPrefillState(FloatArray q, FloatArray keyCache, FloatArray valueCache, 
                             FloatArray output, int nHeads, int headSize, int kvDim, 
                             int kvMul, int position, int contextLength) {
        this.q = q;
        this.keyCache = keyCache;
        this.valueCache = valueCache;
        this.output = output;
        this.nHeads = nHeads;
        this.headSize = headSize;
        this.kvDim = kvDim;
        this.kvMul = kvMul;
        this.position = position;
        this.contextLength = contextLength;
    }
    
    // ========== UTILITY METHODS ==========
    
    /**
     * Validates that all required fields are properly initialized
     * 
     * @return true if state is valid for GPU processing
     */
    public boolean isValid() {
        return q != null && keyCache != null && valueCache != null && output != null &&
               nHeads > 0 && headSize > 0 && kvDim > 0 && kvMul > 0 && 
               position >= 0 && contextLength > 0;
    }
    
    /**
     * Returns a string representation for debugging
     */
    @Override
    public String toString() {
        return String.format("VisionPrefillState{nHeads=%d, headSize=%d, kvDim=%d, kvMul=%d, pos=%d, maxLen=%d, arrays=%s}", 
                           nHeads, headSize, kvDim, kvMul, position, contextLength,
                           (q != null && keyCache != null && valueCache != null && output != null) ? "OK" : "NULL");
    }
}