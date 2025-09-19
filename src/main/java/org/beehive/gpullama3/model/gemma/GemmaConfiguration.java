package org.beehive.gpullama3.model.gemma;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for Gemma 3 models (e.g., Gemma 3 270M).
 * 
 * Gemma models have a unique architecture with a large vocabulary size (256K tokens)
 * resulting in most parameters being in the embedding layer (170M of 270M total).
 */
public record GemmaConfiguration(
    int dim,                    // Model dimension (896 for 270M)
    int hiddenDim,              // FFN hidden dimension (2368 for 270M)
    int numberOfLayers,         // Number of transformer layers (10 for 270M)
    int numberOfHeads,          // Number of attention heads (14 for 270M)
    int numberOfKeyValueHeads,  // Number of KV heads (14 for 270M - full attention)
    int vocabularySize,         // Vocabulary size (256000 - large vocab)
    int contextLength,          // Maximum context length (32768 default)
    float rmsNormEps,           // RMSNorm epsilon (1e-6)
    float ropeTheta             // RoPE theta value (10000.0)
) implements Configuration {

    @Override
    public int numberOfHeadsKey() {
        // Gemma uses standard multi-head attention
        return numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsValue() {
        throw new UnsupportedOperationException("Not supported for gemma.");
    }

    @Override
    public int contextLengthModel() {
        // Return the configured context length
        return contextLength;
    }

    /** Size of each attention head (derived from dim / numberOfHeads) */
    public int headSize() {
        return dim / numberOfHeads;
    }

    /** Key/value dimension (derived from dim * numberOfKeyValueHeads / numberOfHeads) */
    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    /** Multiplier for key/value sharing in multi-query attention */
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length,
     *         or the current instance if newContextLength is negative
     */
    public GemmaConfiguration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new GemmaConfiguration(
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.vocabularySize,
                newContextLength,
                this.rmsNormEps,
                this.ropeTheta
        );
    }

    /**
     * Determines if a layer should use global attention (every 6th layer in Gemma 3).
     * Gemma 3 uses 5:1 local-to-global attention ratio.
     */
    public boolean isGlobalAttentionLayer(int layer) {
        return (layer % 6) == 5;
    }

    /**
     * Returns the local attention window size for Gemma 3 (1,024 tokens).
     */
    public int getLocalWindowSize() {
        return 1024;
    }

    /**
     * Returns enhanced RoPE theta for global layers (1M base frequency).
     */
    public float getGlobalRopeTheta() {
        return 1000000.0f;
    }

    /**
     * Returns QK-norm epsilon for numerical stability.
     */
    public float getQKNormEpsilon() {
        return rmsNormEps;
    }
}