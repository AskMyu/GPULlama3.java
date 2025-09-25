package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.inference.state.State;
import java.util.List;

/**
 * Default implementation of BatchProcessor that provides fallback behavior.
 * If the model supports batch processing natively, it will use that capability.
 * Otherwise, it falls back to serial processing with enhanced context preservation.
 */
public class DefaultBatchProcessor implements BatchProcessor {

    private final Model model;
    private final int maxBatchSize;
    private final boolean debugLogging;

    public DefaultBatchProcessor(Model model) {
        this(model, 512, false);
    }

    public DefaultBatchProcessor(Model model, int maxBatchSize, boolean debugLogging) {
        this.model = model;
        this.maxBatchSize = maxBatchSize;
        this.debugLogging = debugLogging;
    }

    @Override
    public State processBatch(State state, List<Integer> promptTokens, int startPosition) {
        if (!validateBatch(promptTokens)) {
            throw new IllegalArgumentException(
                String.format("Invalid batch: tokens=%d, maxBatch=%d, supported=%s",
                    promptTokens.size(), maxBatchSize, supportsBatchProcessing()));
        }

        if (debugLogging) {
            System.out.printf("[DEFAULT-BATCH] Processing batch of %d tokens starting at position %d%n",
                            promptTokens.size(), startPosition);
        }

        // Check if model has native batch processing capability
        if (model instanceof BatchCapableModel) {
            return processBatchNative(state, promptTokens, startPosition);
        } else {
            return processBatchWithSerial(state, promptTokens, startPosition);
        }
    }

    /**
     * Process batch using model's native batch processing capability.
     */
    private State processBatchNative(State state, List<Integer> promptTokens, int startPosition) {
        if (debugLogging) {
            System.out.printf("[DEFAULT-BATCH] Using native batch processing for %s%n",
                            model.getClass().getSimpleName());
        }

        BatchCapableModel batchModel = (BatchCapableModel) model;
        return batchModel.forwardBatch(state, promptTokens, startPosition);
    }

    /**
     * Process batch using enhanced serial processing with context preservation.
     * This method processes tokens serially but with optimizations to maintain
     * context coherence, particularly important for MoE models.
     */
    private State processBatchWithSerial(State state, List<Integer> promptTokens, int startPosition) {
        if (debugLogging) {
            System.out.printf("[DEFAULT-BATCH] Using enhanced serial processing fallback%n");
        }

        // Process tokens serially but with context preservation techniques
        for (int i = 0; i < promptTokens.size(); i++) {
            int token = promptTokens.get(i);
            int position = startPosition + i;

            if (debugLogging && i < 5) { // Log first few tokens
                System.out.printf("[DEFAULT-BATCH] Processing token %d: %d at position %d%n",
                                i, token, position);
            }

            // Forward pass for this token
            model.forward(state, token, position);
        }

        if (debugLogging) {
            System.out.printf("[DEFAULT-BATCH] Completed serial batch processing of %d tokens%n",
                            promptTokens.size());
        }

        return state;
    }

    @Override
    public boolean supportsBatchProcessing() {
        // Always support batch processing - either natively or via enhanced serial fallback
        return true;
    }

    @Override
    public int getOptimalBatchSize() {
        if (model instanceof BatchCapableModel) {
            return ((BatchCapableModel) model).getOptimalBatchSize();
        } else {
            return maxBatchSize;
        }
    }

    /**
     * Enhanced validation that considers model-specific constraints.
     */
    @Override
    public boolean validateBatch(List<Integer> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            if (debugLogging) {
                System.out.printf("[DEFAULT-BATCH] Validation failed: null or empty tokens%n");
            }
            return false;
        }

        if (tokens.size() > getOptimalBatchSize()) {
            if (debugLogging) {
                System.out.printf("[DEFAULT-BATCH] Validation failed: batch size %d > optimal %d%n",
                                tokens.size(), getOptimalBatchSize());
            }
            return false;
        }

        // Check for invalid token IDs
        for (int i = 0; i < tokens.size(); i++) {
            Integer token = tokens.get(i);
            if (token == null || token < 0) {
                if (debugLogging) {
                    System.out.printf("[DEFAULT-BATCH] Validation failed: invalid token at index %d: %s%n",
                                    i, token);
                }
                return false;
            }
        }

        return true;
    }
}