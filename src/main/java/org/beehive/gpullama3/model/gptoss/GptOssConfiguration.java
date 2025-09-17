package org.beehive.gpullama3.model.gptoss;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for OpenAI GPT-OSS models (e.g., GPT-OSS 20B).
 * 
 * GPT-OSS uses a Mixture-of-Experts (MoE) architecture with:
 * - 32 experts total, 4 active per token
 * - Sparse computation for efficiency
 * - MXFP4 quantization for memory efficiency
 * - Grouped multi-query attention
 */
public record GptOssConfiguration(
    int dim,                    // Model dimension
    int hiddenDim,              // Expert hidden dimension
    int numberOfLayers,         // Number of transformer layers (24 for 20B)
    int numberOfHeads,          // Number of attention heads
    int numberOfKeyValueHeads,  // Number of KV heads (GQA with group size 8)
    int vocabularySize,         // Vocabulary size
    int contextLength,          // Maximum context length (131072 / 128K)
    float rmsNormEps,          // RMSNorm epsilon
    float ropeTheta,           // RoPE theta value
    // MoE specific parameters
    int numExperts,            // Total number of experts (32)
    int activeExperts          // Active experts per token (4)
) implements Configuration {

    @Override
    public int numberOfHeadsKey() {
        return numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }

    /** Size of each attention head */
    public int headSize() {
        return dim / numberOfHeads;
    }

    /** Key/value dimension with GQA */
    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    /** KV multiplier for grouped attention */
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /** Total parameters across all experts */
    public long totalMoEParameters() {
        return (long) numExperts * hiddenDim * dim;
    }

    /** Active parameters per token */
    public long activeParameters() {
        return (long) activeExperts * hiddenDim * dim;
    }

    /**
     * Creates a new Configuration with a different context length.
     */
    public GptOssConfiguration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this;
        }
        return new GptOssConfiguration(
                this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads,
                this.numberOfKeyValueHeads, this.vocabularySize, newContextLength,
                this.rmsNormEps, this.ropeTheta, this.numExperts, this.activeExperts
        );
    }
}