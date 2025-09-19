package org.beehive.gpullama3.model.phi4;

import org.beehive.gpullama3.model.Configuration;

// @formatter:off
public record Phi4Configuration(int dim,
                                int hiddenDim,
                                int numberOfLayers,
                                int numberOfHeads,
                                int numberOfKeyValueHeads,
                                int vocabularySize,
                                int contextLength,
                                float rmsNormEps,
                                float ropeTheta,
                                // Phi-4-Mini specific parameters
                                boolean isReasoningModel,
                                boolean sharedEmbedding) implements Configuration {

    @Override
    public int numberOfHeadsKey() {
        // For Phi4, key heads are the same as key-value heads (GQA)
        return numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsValue() {
        throw new UnsupportedOperationException("Not supported for phi4.");
    }

    @Override
    public int headSize() {
        // Calculate head size from dim and numberOfHeads
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        // Calculate key-value dimension for grouped query attention
        return (dim * numberOfKeyValueHeads) / numberOfHeads;
    }

    @Override
    public int kvMul() {
        // Calculate key-value multiplier for grouped query attention
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }
    
    // Phi-4-Mini specific helper methods
    public boolean isReasoningModel() {
        return isReasoningModel;
    }
    
    public boolean hasSharedEmbedding() {
        return sharedEmbedding;
    }
    
    // Static factory method for Phi-4-Mini-Reasoning
    public static Phi4Configuration createPhi4MiniReasoning(int dim, int hiddenDim, 
            int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, 
            int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
        return new Phi4Configuration(dim, hiddenDim, numberOfLayers, numberOfHeads, 
                numberOfKeyValueHeads, vocabularySize, contextLength, rmsNormEps, 
                ropeTheta, true, true); // Reasoning model with shared embedding
    }
    
    // Static factory method for regular Phi-4
    public static Phi4Configuration createPhi4(int dim, int hiddenDim, 
            int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, 
            int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
        return new Phi4Configuration(dim, hiddenDim, numberOfLayers, numberOfHeads, 
                numberOfKeyValueHeads, vocabularySize, contextLength, rmsNormEps, 
                ropeTheta, false, true); // Regular model with shared embedding
    }
}