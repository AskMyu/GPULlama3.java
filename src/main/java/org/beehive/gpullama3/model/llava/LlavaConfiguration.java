package org.beehive.gpullama3.model.llava;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;

/**
 * Configuration for LLaVA (Large Language and Vision Assistant) models.
 * Extends Llama configuration with vision-specific parameters.
 * 
 * LLaVA Architecture:
 * - Vision Encoder: CLIP-ViT-Large-patch14-336 (1024-dimensional features) 
 * - Language Model: Llama-3-8B-Instruct (4096-dimensional embeddings)
 * - Vision-Language Connector: 2-layer MLP projector (1024 -> 4096 dimensions)
 */
public record LlavaConfiguration(
    // Base Llama configuration (matching LlamaConfiguration parameter names)
    int dim,
    int hiddenDim,
    int numberOfLayers,
    int numberOfHeads,
    int numberOfKeyValueHeads,
    int vocabularySize,
    int contextLength,
    float rmsNormEps,
    float ropeTheta,
    
    // Vision-specific parameters
    int visionTokenCount,        // Number of vision tokens (576 for CLIP-ViT-Large-336)
    String visionProjectorPath,  // Path to mmproj GGUF file
    String languageModelPath,    // Path to language model GGUF file
    boolean isQuantizedInt4      // Whether using INT4 quantization
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
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        return (dim * numberOfKeyValueHeads) / numberOfHeads;
    }

    /**
     * Builder for creating LlavaConfiguration instances.
     */
    public static class Builder {
        private int dim;
        private int hiddenDim;
        private int numberOfLayers;
        private int numberOfHeads;
        private int numberOfKeyValueHeads;
        private int vocabularySize;
        private int contextLength;
        private float rmsNormEps;
        private float ropeTheta;
        
        private int visionTokenCount = 576; // Default for CLIP-ViT-Large-336
        private String visionProjectorPath;
        private String languageModelPath;
        private boolean isQuantizedInt4 = false;

        public Builder fromLlamaConfig(LlamaConfiguration llamaConfig) {
            this.dim = llamaConfig.dim();
            this.hiddenDim = llamaConfig.hiddenDim();
            this.numberOfLayers = llamaConfig.numberOfLayers();
            this.numberOfHeads = llamaConfig.numberOfHeads();
            this.numberOfKeyValueHeads = llamaConfig.numberOfKeyValueHeads();
            this.vocabularySize = llamaConfig.vocabularySize();
            this.contextLength = llamaConfig.contextLength();
            this.rmsNormEps = llamaConfig.rmsNormEps();
            this.ropeTheta = llamaConfig.ropeTheta();
            return this;
        }

        public Builder visionTokenCount(int visionTokenCount) {
            this.visionTokenCount = visionTokenCount;
            return this;
        }

        public Builder visionProjectorPath(String path) {
            this.visionProjectorPath = path;
            return this;
        }

        public Builder languageModelPath(String path) {
            this.languageModelPath = path;
            return this;
        }

        public Builder isQuantizedInt4(boolean quantized) {
            this.isQuantizedInt4 = quantized;
            return this;
        }

        public LlavaConfiguration build() {
            return new LlavaConfiguration(
                dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
                vocabularySize, contextLength, rmsNormEps, ropeTheta,
                visionTokenCount, visionProjectorPath, languageModelPath, isQuantizedInt4
            );
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Get the vision encoder input dimension (CLIP features).
     */
    public int getVisionInputDim() {
        return 1024; // CLIP-ViT-Large-patch14-336 feature dimension
    }

    /**
     * Get the language model embedding dimension.
     */
    public int getLanguageEmbeddingDim() {
        return dim; // Llama-3-8B: 4096
    }

    /**
     * Get the projector hidden dimension (for MLP layers).
     */
    public int getProjectorHiddenDim() {
        return dim; // Same as language embedding for simplicity
    }

    /**
     * Get the vision encoder type.
     */
    public String getVisionEncoderType() {
        return "clip_vit_large_patch14_336";
    }

    /**
     * Get the expected image resolution.
     */
    public int getImageResolution() {
        return 336; // CLIP-ViT-Large-patch14-336 input size
    }

    /**
     * Get the patch size for vision transformer.
     */
    public int getPatchSize() {
        return 14; // CLIP-ViT-Large-patch14-336 patch size
    }

    /**
     * Calculate the number of patches per side.
     */
    public int getPatchesPerSide() {
        return getImageResolution() / getPatchSize(); // 336/14 = 24
    }

    /**
     * Get the total number of vision patches (excluding class token).
     */
    public int getVisionPatchCount() {
        return getPatchesPerSide() * getPatchesPerSide(); // 24*24 = 576
    }

    // Convenience getters with descriptive names
    public int getVisionTokenCount() {
        return visionTokenCount;
    }

    public String getVisionProjectorPath() {
        return visionProjectorPath;
    }

    public String getLanguageModelPath() {
        return languageModelPath;
    }

    public boolean isQuantizedInt4() {
        return isQuantizedInt4;
    }

    /**
     * Get model description for logging.
     */
    public String getModelDescription() {
        return String.format("LLaVA-Llama-3-8B (%s, %d vision tokens, %dx%d patches)", 
                            isQuantizedInt4 ? "INT4" : "F16",
                            visionTokenCount, 
                            getPatchesPerSide(), 
                            getPatchesPerSide());
    }
}