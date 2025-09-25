package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.config.InferenceConfig;
import org.beehive.gpullama3.config.ProcessingMode;
import org.beehive.gpullama3.model.Model;

/**
 * Factory for creating appropriate inference engines based on configuration.
 * Implements the factory pattern to abstract engine creation and provide
 * clean separation between batch-enabled and serial-only engines.
 */
public class InferenceEngineFactory {

    /**
     * Create an inference engine based on the provided configuration.
     * Since InferenceEngine uses static methods, this factory creates the appropriate
     * batch-enabled wrapper when needed.
     *
     * @param model The model to use for inference
     * @param config Configuration specifying processing modes and parameters
     * @return Appropriate inference engine (BatchInferenceEngine)
     */
    public static BatchInferenceEngine createEngine(Model model, InferenceConfig config) {
        if (config == null) {
            System.out.printf("[ENGINE-FACTORY] No config provided, creating engine with serial config%n");
            config = InferenceConfig.createSerialConfig();
        }

        String modelType = model.getClass().getSimpleName().toLowerCase();
        ProcessingMode mode = config.getProcessingMode(modelType);
        boolean batchEnabled = config.isBatchProcessingEnabled();

        System.out.printf("[ENGINE-FACTORY] Creating engine: model=%s, mode=%s, batchEnabled=%s%n",
                         modelType, mode, batchEnabled);

        // Always return BatchInferenceEngine - it handles both batch and serial modes
        System.out.printf("[ENGINE-FACTORY] ✅ Creating BatchInferenceEngine (handles both batch and serial)%n");
        return new BatchInferenceEngine(model, config);
    }

    /**
     * Create a batch-enabled engine specifically.
     * This method forces creation of a BatchInferenceEngine regardless of configuration.
     *
     * @param model The model to use for inference
     * @param config Configuration for batch processing
     * @return BatchInferenceEngine instance
     */
    public static BatchInferenceEngine createBatchEngine(Model model, InferenceConfig config) {
        if (config == null) {
            config = InferenceConfig.createSerialConfig();
        }

        System.out.printf("[ENGINE-FACTORY] ✅ Force-creating BatchInferenceEngine%n");
        return new BatchInferenceEngine(model, config);
    }

    /**
     * Create a serial-only engine specifically.
     * This method creates a BatchInferenceEngine with serial configuration.
     *
     * @param model The model to use for inference
     * @return BatchInferenceEngine configured for serial processing
     */
    public static BatchInferenceEngine createSerialEngine(Model model) {
        InferenceConfig config = InferenceConfig.createSerialConfig();
        System.out.printf("[ENGINE-FACTORY] ✅ Force-creating serial BatchInferenceEngine%n");
        return new BatchInferenceEngine(model, config);
    }

    /**
     * Create an engine optimized for OLMoE models.
     * This specifically addresses the expert routing context isolation problem.
     *
     * @param model The OLMoE model
     * @return BatchInferenceEngine configured for OLMoE batch processing
     */
    public static BatchInferenceEngine createOLMoEEngine(Model model) {
        InferenceConfig config = InferenceConfig.createOLMoEBatchConfig();
        System.out.printf("[ENGINE-FACTORY] ✅ Creating OLMoE-optimized BatchInferenceEngine%n");
        System.out.printf("[ENGINE-FACTORY] 🎯 CONFIGURED TO SOLVE EXPERT ROUTING CONTEXT ISOLATION%n");
        return new BatchInferenceEngine(model, config);
    }

    /**
     * Determine if batch engine should be used based on model and configuration.
     */
    private static boolean shouldUseBatchEngine(Model model, InferenceConfig config) {
        String modelType = model.getClass().getSimpleName().toLowerCase();
        ProcessingMode mode = config.getProcessingMode(modelType);

        // Batch engine beneficial conditions
        boolean batchModeRequested = mode == ProcessingMode.BATCH;
        boolean batchGloballyEnabled = config.isBatchProcessingEnabled();
        boolean modelBenefitsFromBatch = modelType.contains("olmoe"); // OLMoE benefits most

        System.out.printf("[ENGINE-FACTORY] Batch decision: mode=%s, enabled=%s, benefits=%s%n",
                         batchModeRequested, batchGloballyEnabled, modelBenefitsFromBatch);

        return batchModeRequested && batchGloballyEnabled && modelBenefitsFromBatch;
    }

    /**
     * Create engine with automatic model type detection and optimal configuration.
     *
     * @param model The model to analyze and create engine for
     * @return Optimally configured inference engine
     */
    public static BatchInferenceEngine createOptimalEngine(Model model) {
        String modelType = model.getClass().getSimpleName().toLowerCase();

        if (modelType.contains("olmoe")) {
            // OLMoE models benefit from batch processing to solve context isolation
            System.out.printf("[ENGINE-FACTORY] 🎯 Detected OLMoE model - using batch processing%n");
            return createOLMoEEngine(model);
        } else {
            // Other models work well with serial processing
            System.out.printf("[ENGINE-FACTORY] Detected %s model - using serial processing%n", modelType);
            return createSerialEngine(model);
        }
    }

    /**
     * Validate that an engine can be created for the given model and configuration.
     *
     * @param model The model to validate
     * @param config The configuration to validate
     * @return true if a valid engine can be created
     */
    public static boolean validateEngineCreation(Model model, InferenceConfig config) {
        if (model == null) {
            System.err.printf("[ENGINE-FACTORY] ❌ Validation failed: null model%n");
            return false;
        }

        if (config != null && config.isBatchProcessingEnabled()) {
            // Additional validation for batch processing
            try {
                BatchInferenceEngine testEngine = new BatchInferenceEngine(model, config);
                boolean valid = testEngine.validateBatchSetup();
                testEngine.cleanup();

                if (!valid) {
                    System.err.printf("[ENGINE-FACTORY] ❌ Validation failed: batch setup invalid%n");
                }
                return valid;
            } catch (Exception e) {
                System.err.printf("[ENGINE-FACTORY] ❌ Validation failed: %s%n", e.getMessage());
                return false;
            }
        }

        System.out.printf("[ENGINE-FACTORY] ✅ Validation passed%n");
        return true;
    }

    /**
     * Get recommended configuration for a specific model type.
     *
     * @param model The model to get configuration for
     * @return Recommended InferenceConfig
     */
    public static InferenceConfig getRecommendedConfig(Model model) {
        String modelType = model.getClass().getSimpleName().toLowerCase();

        if (modelType.contains("olmoe")) {
            System.out.printf("[ENGINE-FACTORY] 🎯 Recommending batch config for OLMoE%n");
            return InferenceConfig.createOLMoEBatchConfig();
        } else {
            System.out.printf("[ENGINE-FACTORY] Recommending serial config for %s%n", modelType);
            return InferenceConfig.createSerialConfig();
        }
    }

    /**
     * Create engine with debug logging enabled for troubleshooting.
     *
     * @param model The model to use
     * @param enableBatch Whether to enable batch processing
     * @return Inference engine with debug logging
     */
    public static BatchInferenceEngine createDebugEngine(Model model, boolean enableBatch) {
        InferenceConfig config;

        if (enableBatch) {
            config = InferenceConfig.createOLMoEBatchConfig();
        } else {
            config = InferenceConfig.createSerialConfig();
        }

        config.setDebugLogging(true);

        System.out.printf("[ENGINE-FACTORY] 🐛 Creating debug engine: batch=%s%n", enableBatch);

        return createEngine(model, config);
    }
}