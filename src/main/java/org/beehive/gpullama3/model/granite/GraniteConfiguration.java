package org.beehive.gpullama3.model.granite;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for IBM Granite 3.3 models (e.g., Granite 3.3 2B).
 * 
 * Granite models feature a modern dense transformer architecture with:
 * - Grouped Query Attention (GQA) for efficiency
 * - SwiGLU activation function for better performance
 * - Fill-in-the-Middle (FIM) support for code completion
 * - Large context window (128K tokens)
 */
public record GraniteConfiguration(
    int dim,                    // Model dimension (2048 for 2B)
    int hiddenDim,              // FFN hidden dimension (5504 for 2B with SwiGLU)
    int numberOfLayers,         // Number of transformer layers (24 for 2B)
    int numberOfHeads,          // Number of attention heads (16 for 2B)
    int numberOfKeyValueHeads,  // Number of KV heads for GQA (4 for 2B)
    int vocabularySize,         // Vocabulary size (~49159 for Granite models)
    int contextLength,          // Maximum context length (131072 / 128K)
    float rmsNormEps,           // RMSNorm epsilon (1e-5)
    float ropeTheta             // RoPE theta value (10000.0)
) implements Configuration {

    @Override
    public int numberOfHeadsKey() {
        // Granite uses GQA with separate KV head count
        return numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsValue() {
        throw new UnsupportedOperationException("Not supported for granite.");
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

    /** Multiplier for key/value sharing in grouped-query attention */
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
    public GraniteConfiguration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new GraniteConfiguration(
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
     * Checks if this configuration supports Fill-in-the-Middle (FIM) operations.
     * Granite 3.3 models support FIM for code completion tasks.
     */
    public boolean supportsFIM() {
        return true;
    }

    /**
     * Checks if this configuration uses SwiGLU activation.
     * All Granite 3.3 models use SwiGLU instead of standard GELU.
     */
    public boolean usesSwiGLU() {
        return true;
    }
}