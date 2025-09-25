package org.beehive.gpullama3.model.olmoe;

import org.beehive.gpullama3.inference.BatchCapableModel;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;

import java.util.List;

/**
 * Simplified OLMoE batch processor that solves expert routing context isolation
 * without complex GPU batch operations. This uses sequential processing with
 * the proven forward() method to maintain expert routing coherence.
 */
public class OLMoEBatchProcessorSimple implements BatchCapableModel {

    private final Olmoe model;
    private final OlmoeConfiguration config;
    private final Weights weights;
    private final boolean debugLogging;
    private final int maxBatchSize;

    public OLMoEBatchProcessorSimple(Olmoe model, OlmoeConfiguration config, Weights weights,
                                   boolean debugLogging, int maxBatchSize) {
        this.model = model;
        this.config = config;
        this.weights = weights;
        this.debugLogging = debugLogging;
        this.maxBatchSize = maxBatchSize;
    }

    @Override
    public State forwardBatch(State state, List<Integer> tokens, int startPosition) {
        if (!(state instanceof org.beehive.gpullama3.inference.state.OlmoeState)) {
            throw new IllegalArgumentException("OLMoE batch processor requires OlmoeState");
        }

        int batchSize = tokens.size();

        if (debugLogging) {
            System.out.printf("[OLMOE-BATCH] 🚀 Processing batch of %d tokens starting at position %d%n",
                            batchSize, startPosition);
            System.out.printf("[OLMOE-BATCH] 🎯 SOLVING EXPERT ROUTING CONTEXT ISOLATION%n");
        }

        try {
            // CRITICAL FIX: EXPERT CONSISTENCY BATCH PROCESSING
            // Enable expert consistency mode and process all tokens with shared experts

            if (!(state instanceof org.beehive.gpullama3.inference.state.OlmoeState)) {
                throw new RuntimeException("OLMoE batch processor requires OlmoeState");
            }

            if (tokens == null || tokens.isEmpty()) {
                throw new RuntimeException("Cannot process empty token list");
            }

            org.beehive.gpullama3.inference.state.OlmoeState olmoeState =
                (org.beehive.gpullama3.inference.state.OlmoeState) state;

            System.err.printf("[OLMOE-BATCH] 🎯 ENABLING EXPERT CONSISTENCY MODE FOR %d TOKENS%n", tokens.size());

            // STEP 1: Enable expert consistency mode
            olmoeState.batchExpertConsistencyMode = true;

            // STEP 2: Clear any existing shared experts (reset for new batch)
            for (int layer = 0; layer < model.configuration().numberOfLayers(); layer++) {
                olmoeState.sharedExpertsEstablished[layer] = false;
                for (int k = 0; k < model.configuration().numberOfActiveExperts(); k++) {
                    olmoeState.sharedExpertsPerLayer[layer][k] = 0;
                    olmoeState.sharedExpertWeightsPerLayer[layer][k] = 0.0f;
                }
            }

            // STEP 3: Process tokens - first token establishes experts, rest use shared experts
            for (int i = 0; i < tokens.size(); i++) {
                int token = tokens.get(i);
                int position = startPosition + i;

                if (debugLogging) {
                    System.out.printf("[OLMOE-BATCH] Processing token %d/%d: %d at position %d%n",
                                    i + 1, tokens.size(), token, position);
                }

                if (i == 0) {
                    System.err.printf("[OLMOE-BATCH] 🔧 Token 1: ESTABLISHING shared experts for all layers%n");
                } else {
                    System.err.printf("[OLMOE-BATCH] 🔧 Token %d: USING shared experts for consistency%n", i + 1);
                }

                // Process token - the OLMoEGPUProcessor will handle expert consistency automatically
                model.forward(state, token, position);
            }

            // STEP 4: Disable expert consistency mode after batch processing
            olmoeState.batchExpertConsistencyMode = false;

            System.err.printf("[OLMOE-BATCH] ✅ EXPERT CONSISTENCY BATCH PROCESSING COMPLETED%n");

            if (debugLogging) {
                System.out.printf("[OLMOE-BATCH] ✅ Processed %d tokens with expert routing context preservation%n",
                                tokens.size());
            }

        } catch (Exception e) {
            System.err.printf("[OLMOE-BATCH] ❌ Batch processing failed: %s%n", e.getMessage());
            if (debugLogging) {
                e.printStackTrace();
            }
            throw new RuntimeException("OLMoE batch processing failed", e);
        }

        // Update final state
        if (!tokens.isEmpty()) {
            state.latestToken = tokens.get(tokens.size() - 1);
        }

        return state;
    }

    public boolean supportsBatchProcessing() {
        return true;
    }

    @Override
    public boolean isBatchProcessingAvailable() {
        return true;
    }

    @Override
    public int getOptimalBatchSize() {
        return Math.min(maxBatchSize, 64);
    }

    @Override
    public void cleanupBatchProcessing() {
        // No GPU arrays to cleanup in this simple implementation
        if (debugLogging) {
            System.out.printf("[OLMOE-BATCH] ✅ Cleanup completed%n");
        }
    }
}