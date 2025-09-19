package org.beehive.gpullama3.model.deepseekr1;

import org.beehive.gpullama3.attention.mla.MLAConfiguration;
import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for DeepSeek-R1 models with MoE and MLA support.
 *
 * Defines the architecture for the full 671B parameter model with 256 experts per layer
 * and Multi-head Latent Attention. This is completely isolated from existing configurations.
 */
public record DeepSeekR1Configuration(
    // Base transformer parameters
    int dim,                      // Model dimension (embedding size)
    int hiddenDim,                // FFN hidden dimension (base)
    int numberOfLayers,           // Number of transformer layers
    int numberOfHeads,            // Number of attention heads
    int numberOfKeyValueHeads,    // Number of key-value heads (for GQA)
    int vocabSize,                // Vocabulary size
    int contextLengthModel,       // Maximum context length
    int contextLength,            // Current context length
    boolean useParallelResidual,  // Use parallel residual connections
    float rmsNormEps,             // RMS norm epsilon
    float ropeTheta,              // RoPE frequency base

    // MoE-specific parameters
    int totalExperts,             // Total experts per layer (256 for DeepSeek-R1)
    int activeExperts,            // Active experts per token (typically 8-16)
    int expertHiddenDim,          // Expert-specific hidden dimension
    long totalParameters,         // Total model parameters (671B)
    long activeParameters,        // Active parameters per forward pass (37B)
    boolean enableLoadBalancing,  // Apply load balancing to expert routing
    float routingNoise,           // Routing noise during training (0.0 for inference)

    // MLA-specific parameters
    boolean enableMLA,            // Use Multi-head Latent Attention
    MLAConfiguration mlaConfig,   // MLA configuration

    // DeepSeek-R1 specific features
    boolean enableFP8,            // Use FP8 quantization
    boolean conservativeSampling, // Use conservative sampling for reasoning
    String activationFunction     // Expert activation function ("swiglu")
) implements Configuration {

    /**
     * Create configuration for full DeepSeek-R1 671B model.
     */
    public static DeepSeekR1Configuration createFull() {
        // DeepSeek-R1 architecture specifications
        int dim = 7168;              // Model dimension
        int hiddenDim = 18432;       // Base FFN hidden dimension
        int numberOfLayers = 61;     // Total transformer layers
        int numberOfHeads = 56;      // Attention heads
        int numberOfKeyValueHeads = 8; // KV heads (for memory efficiency)
        int vocabSize = 128256;      // Extended vocabulary
        int contextLength = 128000;  // Maximum context length

        // MoE parameters (DeepSeek-R1 specific)
        int totalExperts = 256;      // Experts per layer
        int activeExperts = 8;       // Active experts per token
        int expertHiddenDim = 4096;  // Expert hidden dimension
        long totalParameters = 671_000_000_000L;  // 671B total
        long activeParameters = 37_000_000_000L;  // 37B active

        // MLA configuration for memory efficiency
        MLAConfiguration mlaConfig = MLAConfiguration.forDeepSeekR1(
            dim / numberOfHeads,  // headDim = 128
            numberOfHeads,        // 56 heads
            numberOfKeyValueHeads // 8 KV heads
        );

        return new DeepSeekR1Configuration(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabSize, contextLength, contextLength, false, 1e-6f, 10000.0f,
            totalExperts, activeExperts, expertHiddenDim,
            totalParameters, activeParameters, true, 0.0f,
            true, mlaConfig,
            true, true, "swiglu"
        );
    }

    /**
     * Create configuration for DeepSeek-R1-Distill 7B model.
     */
    public static DeepSeekR1Configuration createDistill7B() {
        // Smaller distilled model
        int dim = 4096;
        int hiddenDim = 11008;
        int numberOfLayers = 32;
        int numberOfHeads = 32;
        int numberOfKeyValueHeads = 8;
        int vocabSize = 128256;
        int contextLength = 32768;

        // No MoE for distilled models
        int totalExperts = 1;        // Single expert (standard FFN)
        int activeExperts = 1;
        int expertHiddenDim = hiddenDim;
        long totalParameters = 7_000_000_000L;   // 7B total
        long activeParameters = 7_000_000_000L;  // All active

        // MLA still beneficial for distilled models
        MLAConfiguration mlaConfig = MLAConfiguration.forDeepSeekR1(
            dim / numberOfHeads,  // headDim = 128
            numberOfHeads,
            numberOfKeyValueHeads
        );

        return new DeepSeekR1Configuration(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabSize, contextLength, contextLength, false, 1e-6f, 10000.0f,
            totalExperts, activeExperts, expertHiddenDim,
            totalParameters, activeParameters, false, 0.0f,
            true, mlaConfig,
            false, true, "swiglu"
        );
    }

    /**
     * Validate configuration parameters.
     */
    public DeepSeekR1Configuration {
        // Basic validation
        if (dim <= 0) throw new IllegalArgumentException("Model dimension must be positive");
        if (numberOfLayers <= 0) throw new IllegalArgumentException("Number of layers must be positive");
        if (numberOfHeads <= 0) throw new IllegalArgumentException("Number of heads must be positive");
        if (numberOfKeyValueHeads <= 0 || numberOfKeyValueHeads > numberOfHeads) {
            throw new IllegalArgumentException("Invalid number of KV heads");
        }

        // MoE validation (only for MoE models)
        if (totalExperts > 1) {
            // This is a MoE model - validate MoE parameters
            if (activeExperts <= 0 || activeExperts > totalExperts) {
                throw new IllegalArgumentException("Invalid number of active experts");
            }
        } else {
            // Non-MoE model - set sensible defaults
            if (totalExperts < 0) throw new IllegalArgumentException("Total experts cannot be negative");
        }
        if (totalParameters <= 0) throw new IllegalArgumentException("Total parameters must be positive");
        if (activeParameters <= 0 || activeParameters > totalParameters) {
            throw new IllegalArgumentException("Invalid active parameters");
        }

        // MLA validation
        if (enableMLA && mlaConfig == null) {
            throw new IllegalArgumentException("MLA enabled but no configuration provided");
        }

        // Head dimension must be divisible
        if (dim % numberOfHeads != 0) {
            throw new IllegalArgumentException("Model dimension must be divisible by number of heads");
        }
    }

    /**
     * Calculate head dimension.
     */
    public int headSize() {
        return dim / numberOfHeads;
    }

    /**
     * Get vocabulary size (required by Configuration interface).
     */
    public int vocabularySize() {
        return vocabSize;
    }

    /**
     * Get number of heads for keys (required by Configuration interface).
     */
    public int numberOfHeadsKey() {
        return numberOfKeyValueHeads;
    }

    /**
     * Get number of heads for values (required by Configuration interface).
     */
    public int numberOfHeadsValue() {
        return numberOfKeyValueHeads;
    }

    /**
     * Calculate KV dimension (for GQA).
     */
    public int kvDim() {
        return dim / numberOfHeads; // Same as head size in DeepSeek-R1
    }

    /**
     * Calculate KV multiplier for GQA.
     */
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /**
     * Check if this is a full MoE model.
     */
    public boolean isMoEModel() {
        return totalExperts > 1;
    }

    /**
     * Check if this is a distilled model.
     */
    public boolean isDistilledModel() {
        return totalParameters < 100_000_000_000L; // Less than 100B = distilled
    }

    /**
     * Get expert sparsity ratio.
     */
    public float getExpertSparsity() {
        return (float) activeExperts / totalExperts;
    }

    /**
     * Get parameter efficiency (active/total).
     */
    public float getParameterEfficiency() {
        return (float) activeParameters / totalParameters;
    }

    /**
     * Get memory efficiency from MLA.
     */
    public float getMLAMemoryReduction() {
        return enableMLA ? mlaConfig.getMemoryReductionFactor() : 0.0f;
    }

    /**
     * Estimate total memory usage for inference.
     */
    public long estimateMemoryUsage(int batchSize, int seqLen) {
        // Model weights
        long modelMemory = activeParameters * 2L; // FP16

        // KV cache
        long kvCacheMemory;
        if (enableMLA) {
            kvCacheMemory = mlaConfig.estimateKVCacheMemory(seqLen, batchSize);
        } else {
            kvCacheMemory = 2L * batchSize * seqLen * numberOfKeyValueHeads * headSize() * 4L;
        }

        // Activations (rough estimate)
        long activationMemory = batchSize * seqLen * dim * 4L;

        return modelMemory + kvCacheMemory + activationMemory;
    }

    /**
     * Create a copy with different context length.
     */
    public DeepSeekR1Configuration withContextLength(int newContextLength) {
        return new DeepSeekR1Configuration(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabSize, contextLengthModel, newContextLength, useParallelResidual,
            rmsNormEps, ropeTheta, totalExperts, activeExperts, expertHiddenDim,
            totalParameters, activeParameters, enableLoadBalancing, routingNoise,
            enableMLA, mlaConfig, enableFP8, conservativeSampling, activationFunction
        );
    }
}