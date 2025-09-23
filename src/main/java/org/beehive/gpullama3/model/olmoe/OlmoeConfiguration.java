package org.beehive.gpullama3.model.olmoe;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for OLMoE-1B-7B model.
 * 
 * OLMoE is a Mixture-of-Experts model with 64 experts total, 8 active per token.
 * Total parameters: 7B, Active parameters: 1B per token.
 */
// @formatter:off
public record OlmoeConfiguration(int dim,                      // Hidden size: 2048
                                 int hiddenDim,                 // Intermediate size per expert: 1024
                                 int numberOfLayers,            // 16 layers
                                 int numberOfHeads,             // 16 attention heads
                                 int numberOfKeyValueHeads,     // 16 (no GQA)
                                 int vocabularySize,            // 50,304 tokens (OLMo vocabulary)
                                 int contextLength,             // 4096 tokens
                                 int numberOfExperts,           // 64 experts total
                                 int numberOfActiveExperts,     // 8 experts active per token (Top-8)
                                 float rmsNormEps,              // 1e-05
                                 float ropeTheta,               // 10000.0
                                 float routerAuxLossCoef,       // 0.01 for load balancing
                                 boolean outputRouterLogits,    // false by default
                                 float finalLogitSoftcapping)   // CRITICAL: Final logit softcapping factor
                                 implements Configuration {

    @Override
    public int numberOfHeadsKey() {
        return numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsValue() {
        throw new UnsupportedOperationException("Not supported for olmoe.");
    }

    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        // OLMoE doesn't use GQA, so kvDim equals dim
        return dim;
    }

    @Override
    public int kvMul() {
        // No key-value sharing in OLMoE
        return 1;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }
    
    // MoE-specific helper methods
    public boolean isMoEModel() {
        return true; // OLMoE is always MoE
    }
    
    public int getNumberOfExperts() {
        return numberOfExperts;
    }
    
    public int getNumberOfActiveExperts() {
        return numberOfActiveExperts;
    }
    
    public float getRouterAuxLossCoef() {
        return routerAuxLossCoef;
    }
    
    public boolean shouldOutputRouterLogits() {
        return outputRouterLogits;
    }

    public float getFinalLogitSoftcapping() {
        return finalLogitSoftcapping;
    }
    
    // Static factory method for OLMoE-1B-7B
    public static OlmoeConfiguration createOLMoE1B7B() {
        return new OlmoeConfiguration(
            2048,      // dim (hidden size)
            1024,      // hiddenDim (intermediate size per expert)
            16,        // numberOfLayers
            16,        // numberOfHeads
            16,        // numberOfKeyValueHeads (no GQA)
            50304,     // vocabularySize (OLMo tokenizer)
            4096,      // contextLength
            64,        // numberOfExperts
            8,         // numberOfActiveExperts (Top-8)
            1e-05f,    // rmsNormEps
            10000.0f,  // ropeTheta
            0.01f,     // routerAuxLossCoef
            false,     // outputRouterLogits
            30.0f      // CRITICAL: finalLogitSoftcapping (matches llama.cpp default)
        );
    }
    
    // Static factory method for custom OLMoE configurations
    public static OlmoeConfiguration create(int dim, int hiddenDim, int numberOfLayers,
            int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize,
            int contextLength, int numberOfExperts, int numberOfActiveExperts,
            float rmsNormEps, float ropeTheta, float routerAuxLossCoef,
            boolean outputRouterLogits, float finalLogitSoftcapping) {
        return new OlmoeConfiguration(dim, hiddenDim, numberOfLayers, numberOfHeads,
                numberOfKeyValueHeads, vocabularySize, contextLength, numberOfExperts,
                numberOfActiveExperts, rmsNormEps, ropeTheta, routerAuxLossCoef,
                outputRouterLogits, finalLogitSoftcapping);
    }
}
// @formatter:on