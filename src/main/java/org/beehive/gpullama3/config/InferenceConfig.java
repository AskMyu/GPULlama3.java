package org.beehive.gpullama3.config;

import java.util.HashMap;
import java.util.Map;

/**
 * Configuration class for inference processing modes.
 * Supports global and model-specific processing mode configuration.
 */
public class InferenceConfig {
    private ProcessingMode defaultProcessingMode = ProcessingMode.SERIAL;
    private Map<String, ProcessingMode> modelSpecificModes = new HashMap<>();
    private boolean batchProcessingEnabled = false;
    private int maxBatchSize = 512;
    private boolean debugLogging = false;

    public InferenceConfig() {
        // Initialize with safe defaults
        initializeDefaults();
    }

    private void initializeDefaults() {
        // OLMoE benefits from batch processing to solve expert routing context isolation
        modelSpecificModes.put("olmoe", ProcessingMode.BATCH);

        // Other models work well with serial processing
        modelSpecificModes.put("llama3", ProcessingMode.SERIAL);
        modelSpecificModes.put("mistral", ProcessingMode.SERIAL);
        modelSpecificModes.put("qwen", ProcessingMode.SERIAL);
        modelSpecificModes.put("phi3", ProcessingMode.SERIAL);
    }

    /**
     * Get the processing mode for a specific model type.
     *
     * @param modelType The model type identifier (e.g., "olmoe", "llama3")
     * @return The processing mode to use for this model
     */
    public ProcessingMode getProcessingMode(String modelType) {
        if (modelType == null) {
            return defaultProcessingMode;
        }

        String normalizedType = modelType.toLowerCase().trim();
        return modelSpecificModes.getOrDefault(normalizedType, defaultProcessingMode);
    }

    /**
     * Set the default processing mode for all models.
     *
     * @param mode The default processing mode
     */
    public void setDefaultProcessingMode(ProcessingMode mode) {
        this.defaultProcessingMode = mode;
        System.out.printf("[INFERENCE-CONFIG] Default processing mode set to: %s%n", mode);
    }

    /**
     * Set processing mode for a specific model type.
     *
     * @param modelType The model type identifier
     * @param mode The processing mode for this model
     */
    public void setModelSpecificMode(String modelType, ProcessingMode mode) {
        if (modelType != null) {
            String normalizedType = modelType.toLowerCase().trim();
            modelSpecificModes.put(normalizedType, mode);
            System.out.printf("[INFERENCE-CONFIG] Model-specific mode set: %s -> %s%n", normalizedType, mode);
        }
    }

    /**
     * Check if batch processing is enabled globally.
     *
     * @return true if batch processing is enabled
     */
    public boolean isBatchProcessingEnabled() {
        return batchProcessingEnabled;
    }

    /**
     * Enable or disable batch processing globally.
     *
     * @param enabled true to enable batch processing
     */
    public void setBatchProcessingEnabled(boolean enabled) {
        this.batchProcessingEnabled = enabled;
        System.out.printf("[INFERENCE-CONFIG] Batch processing globally %s%n",
                         enabled ? "ENABLED" : "DISABLED");
    }

    /**
     * Get the maximum batch size for batch processing.
     *
     * @return Maximum number of tokens to process in a single batch
     */
    public int getMaxBatchSize() {
        return maxBatchSize;
    }

    /**
     * Set the maximum batch size for batch processing.
     *
     * @param maxBatchSize Maximum batch size (must be > 0)
     */
    public void setMaxBatchSize(int maxBatchSize) {
        if (maxBatchSize > 0) {
            this.maxBatchSize = maxBatchSize;
            System.out.printf("[INFERENCE-CONFIG] Max batch size set to: %d%n", maxBatchSize);
        } else {
            System.err.printf("[INFERENCE-CONFIG] Invalid batch size %d, keeping %d%n",
                            maxBatchSize, this.maxBatchSize);
        }
    }

    /**
     * Check if debug logging is enabled for batch processing.
     *
     * @return true if debug logging is enabled
     */
    public boolean isDebugLogging() {
        return debugLogging;
    }

    /**
     * Enable or disable debug logging for batch processing.
     *
     * @param debugLogging true to enable debug logging
     */
    public void setDebugLogging(boolean debugLogging) {
        this.debugLogging = debugLogging;
        System.out.printf("[INFERENCE-CONFIG] Debug logging %s%n",
                         debugLogging ? "ENABLED" : "DISABLED");
    }

    /**
     * Check if a specific model should use batch processing.
     *
     * @param modelType The model type to check
     * @return true if the model should use batch processing
     */
    public boolean shouldUseBatchProcessing(String modelType) {
        ProcessingMode mode = getProcessingMode(modelType);
        return mode == ProcessingMode.BATCH && batchProcessingEnabled;
    }

    /**
     * Create a configuration with batch processing enabled for OLMoE.
     *
     * @return InferenceConfig with batch processing enabled for OLMoE
     */
    public static InferenceConfig createOLMoEBatchConfig() {
        InferenceConfig config = new InferenceConfig();
        config.setBatchProcessingEnabled(true);
        config.setModelSpecificMode("olmoe", ProcessingMode.BATCH);
        config.setDebugLogging(true);
        System.out.println("[INFERENCE-CONFIG] Created OLMoE batch processing configuration");
        return config;
    }

    /**
     * Create a configuration with serial processing for all models.
     *
     * @return InferenceConfig with serial processing (safe default)
     */
    public static InferenceConfig createSerialConfig() {
        InferenceConfig config = new InferenceConfig();
        config.setDefaultProcessingMode(ProcessingMode.SERIAL);
        config.setBatchProcessingEnabled(false);
        System.out.println("[INFERENCE-CONFIG] Created serial processing configuration");
        return config;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("InferenceConfig{");
        sb.append("defaultMode=").append(defaultProcessingMode);
        sb.append(", batchEnabled=").append(batchProcessingEnabled);
        sb.append(", maxBatchSize=").append(maxBatchSize);
        sb.append(", modelModes=").append(modelSpecificModes);
        sb.append("}");
        return sb.toString();
    }
}