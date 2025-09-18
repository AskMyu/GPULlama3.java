package org.beehive.gpullama3.model.gemma;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration record specifically for Gemma 2 models.
 * Includes Gemma 2-specific parameters like logit soft-capping.
 * SEPARATE from GemmaConfiguration to maintain clean architecture.
 */
public record Gemma2Configuration(
    int dim,                        // 2304 for Gemma 2 2B
    int hiddenDim,                  // 9216 for Gemma 2 2B
    int numberOfLayers,             // 26 for Gemma 2 2B
    int numberOfHeads,              // 8 for Gemma 2 2B
    int numberOfKeyValueHeads,      // 4 for Gemma 2 2B (GQA)
    int vocabularySize,             // 256000
    int contextLength,              // 8192 for Gemma 2
    float rmsNormEps,
    float ropeTheta,
    float finalLogitSoftcapping,    // 30.0 - CRITICAL for oscillation fix
    float attnLogitSoftcapping      // 50.0 - CRITICAL for oscillation fix
) implements Configuration {

    @Override
    public int numberOfHeadsKey() {
        return numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }

    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }


    public int keyValueHeadSize() {
        return dim / numberOfKeyValueHeads;
    }

    /** Size of each attention head (derived from dim / numberOfHeads) */
    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    /** Multiplier for key/value sharing in multi-query attention */
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    // Gemma 2-specific helper methods
    public boolean hasLogitSoftcapping() {
        return finalLogitSoftcapping > 0.0f;
    }

    public boolean hasAttentionSoftcapping() {
        return attnLogitSoftcapping > 0.0f;
    }

    /**
     * Determines if layer should use local or global attention in Gemma 2's interleaved pattern.
     * Gemma 2 alternates: local, global, local, global...
     */
    public boolean isLocalAttentionLayer(int layerIndex) {
        return (layerIndex % 2) == 0; // Even layers = local, odd layers = global
    }

    /**
     * Get local attention window size for Gemma 2.
     * @return 4096 tokens (different from Gemma 3's 1024)
     */
    public int getLocalWindowSize() {
        return 4096;
    }

    /**
     * Get global attention span for Gemma 2.
     * @return 8192 tokens (different from Gemma 3's full context)
     */
    public int getGlobalSpan() {
        return 8192;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length,
     *         or the current instance if newContextLength is negative
     */
    public Gemma2Configuration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new Gemma2Configuration(
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.vocabularySize,
                newContextLength,
                this.rmsNormEps,
                this.ropeTheta,
                this.finalLogitSoftcapping,
                this.attnLogitSoftcapping
        );
    }
}