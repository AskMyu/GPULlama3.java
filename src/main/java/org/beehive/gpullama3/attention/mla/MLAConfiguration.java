package org.beehive.gpullama3.attention.mla;

/**
 * Configuration for Multi-head Latent Attention (MLA) as used in DeepSeek-R1.
 *
 * MLA compresses K and V matrices into latent vectors, achieving significant memory reduction
 * while maintaining attention quality. This is isolated from standard attention mechanisms.
 */
public record MLAConfiguration(
    int headDim,              // Attention head dimension
    int latentDim,            // Compressed latent dimension
    int numHeads,             // Number of attention heads
    int kvHeads,              // Number of key-value heads (for GQA compatibility)
    float compressionRatio,   // Target compression ratio (0.05-0.13)
    boolean enableGPUDecomp,  // GPU-accelerated decompression
    boolean fallbackStandard  // Fallback to standard attention if MLA fails
) {

    /**
     * Create MLA configuration optimized for DeepSeek-R1 architecture.
     */
    public static MLAConfiguration forDeepSeekR1(int headDim, int numHeads, int kvHeads) {
        // DeepSeek-R1 uses aggressive compression to achieve 5-13% memory usage
        int latentDim = Math.max(16, headDim / 8); // 87.5% compression
        float compressionRatio = (float) latentDim / headDim;

        return new MLAConfiguration(
            headDim,
            latentDim,
            numHeads,
            kvHeads,
            compressionRatio,
            true,  // Enable GPU decompression for performance
            true   // Always provide fallback for safety
        );
    }

    /**
     * Validate configuration parameters.
     */
    public MLAConfiguration {
        if (headDim <= 0) throw new IllegalArgumentException("Head dimension must be positive");
        if (latentDim <= 0) throw new IllegalArgumentException("Latent dimension must be positive");
        if (latentDim > headDim) throw new IllegalArgumentException("Latent dimension cannot exceed head dimension");
        if (numHeads <= 0) throw new IllegalArgumentException("Number of heads must be positive");
        if (kvHeads <= 0) throw new IllegalArgumentException("Number of KV heads must be positive");
        if (compressionRatio <= 0 || compressionRatio > 1) {
            throw new IllegalArgumentException("Compression ratio must be in (0, 1]");
        }
    }

    /**
     * Calculate memory reduction factor.
     */
    public float getMemoryReductionFactor() {
        return 1.0f - compressionRatio;
    }

    /**
     * Calculate expected memory usage for KV cache with MLA.
     */
    public long estimateKVCacheMemory(int sequenceLength, int batchSize) {
        // Standard KV cache: 2 * batchSize * sequenceLength * numHeads * headDim * sizeof(float)
        long standardMemory = 2L * batchSize * sequenceLength * kvHeads * headDim * 4;

        // MLA compressed: latentDim replaces headDim
        long mlaMemory = 2L * batchSize * sequenceLength * kvHeads * latentDim * 4;

        return mlaMemory;
    }
}