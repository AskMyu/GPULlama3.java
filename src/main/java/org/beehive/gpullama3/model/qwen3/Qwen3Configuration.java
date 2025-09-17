package org.beehive.gpullama3.model.qwen3;

import org.beehive.gpullama3.model.Configuration;

// @formatter:off
public record Qwen3Configuration(int dim,
                                 int hiddenDim,
                                 int numberOfLayers,
                                 int numberOfHeads,
                                 int numberOfKeyValueHeads,
                                 int numberOfHeadsKey,
                                 int numberOfHeadsValue,
                                 int vocabularySize,
                                 int contextLengthModel,
                                 int contextLength,
                                 boolean sharedWeights,
                                 float rmsNormEps,
                                 float ropeTheta,
                                 // MoE-specific parameters for Qwen3-30B-A3B
                                 int numberOfExperts,
                                 int numberOfActiveExperts,
                                 boolean isMoE) implements Configuration {
    @Override
    public int headSize() {
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    @Override
    public int kvDim() {
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    @Override
    public int kvMul() {
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }
    
    // MoE-specific helper methods
    public boolean isMoEModel() {
        return isMoE;
    }
    
    public int getNumberOfExperts() {
        return numberOfExperts;
    }
    
    public int getNumberOfActiveExperts() {
        return numberOfActiveExperts;
    }
    
    // Static factory method for dense Qwen3 models (backward compatibility)
    public static Qwen3Configuration createDense(int dim, int hiddenDim, int numberOfLayers, 
            int numberOfHeads, int numberOfKeyValueHeads, int numberOfHeadsKey, 
            int numberOfHeadsValue, int vocabularySize, int contextLengthModel, 
            int contextLength, boolean sharedWeights, float rmsNormEps, float ropeTheta) {
        return new Qwen3Configuration(dim, hiddenDim, numberOfLayers, numberOfHeads, 
                numberOfKeyValueHeads, numberOfHeadsKey, numberOfHeadsValue, vocabularySize,
                contextLengthModel, contextLength, sharedWeights, rmsNormEps, ropeTheta,
                0, 0, false); // No MoE for dense models
    }
    
    // Static factory method for MoE Qwen3 models (Qwen3-30B-A3B)
    public static Qwen3Configuration createMoE(int dim, int hiddenDim, int numberOfLayers, 
            int numberOfHeads, int numberOfKeyValueHeads, int numberOfHeadsKey, 
            int numberOfHeadsValue, int vocabularySize, int contextLengthModel, 
            int contextLength, boolean sharedWeights, float rmsNormEps, float ropeTheta,
            int numberOfExperts, int numberOfActiveExperts) {
        return new Qwen3Configuration(dim, hiddenDim, numberOfLayers, numberOfHeads, 
                numberOfKeyValueHeads, numberOfHeadsKey, numberOfHeadsValue, vocabularySize,
                contextLengthModel, contextLength, sharedWeights, rmsNormEps, ropeTheta,
                numberOfExperts, numberOfActiveExperts, true); // MoE enabled
    }
}
