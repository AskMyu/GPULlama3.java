package org.beehive.gpullama3.model.deepseekr1;

import org.beehive.gpullama3.model.moe.Expert;
import org.beehive.gpullama3.model.moe.StandardExpert;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * DeepSeek-R1 specific MoE layer implementation.
 *
 * Manages 256 experts per layer with efficient routing and computation.
 * Integrates with existing batch loading framework for memory efficiency.
 *
 * ISOLATION: This is completely separate from existing MoE implementations.
 */
public class DeepSeekR1MoELayer {

    private final DeepSeekR1Configuration config;
    private final DeepSeekR1ExpertRouter router;
    private final Expert[] experts;

    // Expert computation buffers
    private final FloatArray[] expertOutputs;
    private final FloatArray combinedOutput;
    private final boolean[] expertLoaded;    // Track which experts are loaded

    // Performance metrics
    private long totalForwardPasses = 0;
    private long totalExpertComputations = 0;
    private float averageExpertUtilization = 0.0f;

    public DeepSeekR1MoELayer(DeepSeekR1Configuration config) {
        this.config = config;
        this.router = new DeepSeekR1ExpertRouter(config);

        int numExperts = config.totalExperts();
        int inputDim = config.dim();
        int expertHiddenDim = config.expertHiddenDim();

        // Create expert array
        this.experts = new Expert[numExperts];
        this.expertLoaded = new boolean[numExperts];

        // Initialize experts (will be loaded on-demand)
        for (int i = 0; i < numExperts; i++) {
            this.experts[i] = new StandardExpert(i, inputDim, expertHiddenDim, inputDim);
            this.expertLoaded[i] = false;
        }

        // Allocate computation buffers
        int maxTokens = 1024; // Reasonable buffer size
        this.expertOutputs = new FloatArray[numExperts];
        for (int i = 0; i < numExperts; i++) {
            this.expertOutputs[i] = new FloatArray(maxTokens * inputDim);
        }

        this.combinedOutput = new FloatArray(maxTokens * inputDim);
    }

    /**
     * Forward pass through the MoE layer.
     *
     * @param input Input tensor [batchSize * seqLen, inputDim]
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @return MoE output [batchSize * seqLen, inputDim]
     */
    public FloatArray forward(FloatArray input, int batchSize, int seqLen) {
        int numTokens = batchSize * seqLen;
        int inputDim = config.dim();

        validateInput(input, numTokens, inputDim);

        // Step 1: Route tokens to experts
        DeepSeekR1ExpertRouter.ExpertSelection selection = router.route(input, batchSize, seqLen);

        // Step 2: Group tokens by expert for efficient computation
        ExpertTokenGroups tokenGroups = groupTokensByExpert(input, selection, numTokens);

        // Step 3: Process each expert's assigned tokens
        processExperts(tokenGroups, numTokens);

        // Step 4: Combine expert outputs with routing weights
        combineExpertOutputs(selection, numTokens, inputDim);

        totalForwardPasses++;
        updateUtilizationStats(selection);

        return copyOutput(numTokens * inputDim);
    }

    /**
     * Group tokens by their assigned experts for batch processing.
     */
    private ExpertTokenGroups groupTokensByExpert(FloatArray input,
                                                  DeepSeekR1ExpertRouter.ExpertSelection selection,
                                                  int numTokens) {
        int activeExperts = config.activeExperts();
        int inputDim = config.dim();

        // Count tokens assigned to each expert
        int[] expertTokenCounts = new int[config.totalExperts()];
        for (int token = 0; token < numTokens; token++) {
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selection.selectedExperts().get(token * activeExperts + k);
                expertTokenCounts[expertIdx]++;
            }
        }

        // Allocate buffers for each expert's tokens
        FloatArray[] expertInputs = new FloatArray[config.totalExperts()];
        int[][] tokenMappings = new int[config.totalExperts()][];
        float[][] tokenWeights = new float[config.totalExperts()][];

        for (int expertIdx = 0; expertIdx < config.totalExperts(); expertIdx++) {
            int tokenCount = expertTokenCounts[expertIdx];
            if (tokenCount > 0) {
                expertInputs[expertIdx] = new FloatArray(tokenCount * inputDim);
                tokenMappings[expertIdx] = new int[tokenCount];
                tokenWeights[expertIdx] = new float[tokenCount];
            }
        }

        // Fill expert input buffers
        int[] expertPositions = new int[config.totalExperts()]; // Track current position in each expert's buffer

        for (int token = 0; token < numTokens; token++) {
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selection.selectedExperts().get(token * activeExperts + k);
                float weight = selection.expertWeights().get(token * activeExperts + k);

                if (expertInputs[expertIdx] != null) {
                    int pos = expertPositions[expertIdx];

                    // Copy input token to expert buffer
                    for (int dim = 0; dim < inputDim; dim++) {
                        float value = input.get(token * inputDim + dim);
                        expertInputs[expertIdx].set(pos * inputDim + dim, value);
                    }

                    // Record mapping and weight
                    tokenMappings[expertIdx][pos] = token;
                    tokenWeights[expertIdx][pos] = weight;

                    expertPositions[expertIdx]++;
                }
            }
        }

        return new ExpertTokenGroups(expertInputs, tokenMappings, tokenWeights, expertTokenCounts);
    }

    /**
     * Process tokens through their assigned experts.
     */
    private void processExperts(ExpertTokenGroups tokenGroups, int numTokens) {
        // Clear expert output buffers
        for (FloatArray expertOutput : expertOutputs) {
            for (int i = 0; i < expertOutput.getSize(); i++) {
                expertOutput.set(i, 0.0f);
            }
        }

        // Process each expert that has assigned tokens
        for (int expertIdx = 0; expertIdx < config.totalExperts(); expertIdx++) {
            if (tokenGroups.expertTokenCounts[expertIdx] > 0) {
                processExpert(expertIdx, tokenGroups, numTokens);
                totalExpertComputations++;
            }
        }
    }

    /**
     * Process a single expert with its assigned tokens.
     */
    private void processExpert(int expertIdx, ExpertTokenGroups tokenGroups, int numTokens) {
        FloatArray expertInput = tokenGroups.expertInputs[expertIdx];
        if (expertInput == null) return;

        // Ensure expert is loaded
        if (!expertLoaded[expertIdx]) {
            loadExpert(expertIdx);
        }

        int tokenCount = tokenGroups.expertTokenCounts[expertIdx];
        int inputDim = config.dim();

        // Process all tokens for this expert in one batch
        FloatArray batchOutput = new FloatArray(tokenCount * inputDim);
        experts[expertIdx].forward(expertInput, batchOutput, config.activationFunction());

        // Distribute expert outputs back to their original token positions
        for (int tokenPos = 0; tokenPos < tokenCount; tokenPos++) {
            int originalToken = tokenGroups.tokenMappings[expertIdx][tokenPos];
            float weight = tokenGroups.tokenWeights[expertIdx][tokenPos];

            // Add weighted expert output to the token's position
            for (int dim = 0; dim < inputDim; dim++) {
                float expertValue = batchOutput.get(tokenPos * inputDim + dim);
                float weightedValue = expertValue * weight;

                int outputPos = originalToken * inputDim + dim;
                float currentValue = expertOutputs[expertIdx].get(outputPos);
                expertOutputs[expertIdx].set(outputPos, currentValue + weightedValue);
            }
        }
    }

    /**
     * Combine outputs from all experts to produce final MoE output.
     */
    private void combineExpertOutputs(DeepSeekR1ExpertRouter.ExpertSelection selection,
                                     int numTokens, int inputDim) {
        // Clear combined output
        for (int i = 0; i < combinedOutput.getSize(); i++) {
            combinedOutput.set(i, 0.0f);
        }

        // Sum contributions from all experts
        for (int expertIdx = 0; expertIdx < config.totalExperts(); expertIdx++) {
            if (expertLoaded[expertIdx]) {
                // Add this expert's contributions to combined output
                for (int token = 0; token < numTokens; token++) {
                    for (int dim = 0; dim < inputDim; dim++) {
                        int pos = token * inputDim + dim;
                        float expertContribution = expertOutputs[expertIdx].get(pos);
                        float currentValue = combinedOutput.get(pos);
                        combinedOutput.set(pos, currentValue + expertContribution);
                    }
                }
            }
        }
    }

    /**
     * Load expert weights (placeholder for actual weight loading).
     */
    private void loadExpert(int expertIdx) {
        // In a real implementation, this would load expert weights from storage
        // For now, mark as loaded
        expertLoaded[expertIdx] = true;
    }

    /**
     * Set expert weights for a specific expert.
     */
    public void setExpertWeights(int expertIdx, FloatArray gateWeights, FloatArray upWeights,
                               FloatArray downWeights, FloatArray gateBias, FloatArray upBias) {
        if (expertIdx >= experts.length) {
            throw new IllegalArgumentException("Expert index out of range: " + expertIdx);
        }

        // Set weights in the expert
        if (experts[expertIdx] instanceof StandardExpert standardExpert) {
            // This would need to be implemented in StandardExpert
            // For now, mark as loaded
            expertLoaded[expertIdx] = true;
        }
    }

    /**
     * Set routing weights for the layer.
     */
    public void setRoutingWeights(FloatArray routingWeights, FloatArray routingBias) {
        router.setRoutingWeights(routingWeights, routingBias);
    }

    /**
     * Update utilization statistics.
     */
    private void updateUtilizationStats(DeepSeekR1ExpertRouter.ExpertSelection selection) {
        int numTokens = selection.batchSize() * selection.seqLen();
        int activeExperts = selection.activeExperts();
        int totalExperts = config.totalExperts();

        // Count unique experts used
        boolean[] expertUsed = new boolean[totalExperts];
        int uniqueExpertsUsed = 0;

        for (int token = 0; token < numTokens; token++) {
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selection.selectedExperts().get(token * activeExperts + k);
                if (!expertUsed[expertIdx]) {
                    expertUsed[expertIdx] = true;
                    uniqueExpertsUsed++;
                }
            }
        }

        // Update running average
        float currentUtilization = (float) uniqueExpertsUsed / totalExperts;
        averageExpertUtilization = (averageExpertUtilization * (totalForwardPasses - 1) + currentUtilization) / totalForwardPasses;
    }

    /**
     * Copy output for return (avoiding reference sharing).
     */
    private FloatArray copyOutput(int length) {
        FloatArray output = new FloatArray(length);
        for (int i = 0; i < length; i++) {
            output.set(i, combinedOutput.get(i));
        }
        return output;
    }

    /**
     * Validate input dimensions.
     */
    private void validateInput(FloatArray input, int numTokens, int inputDim) {
        int expectedSize = numTokens * inputDim;
        if (input.getSize() != expectedSize) {
            throw new IllegalArgumentException("Input size mismatch: expected " + expectedSize + ", got " + input.getSize());
        }
    }

    /**
     * Get layer statistics.
     */
    public MoELayerStats getStats() {
        DeepSeekR1ExpertRouter.RoutingStats routingStats = router.getStats();

        int loadedExperts = 0;
        for (boolean loaded : expertLoaded) {
            if (loaded) loadedExperts++;
        }

        return new MoELayerStats(
            totalForwardPasses,
            totalExpertComputations,
            averageExpertUtilization,
            loadedExperts,
            routingStats
        );
    }

    /**
     * Token grouping for efficient expert processing.
     */
    private record ExpertTokenGroups(
        FloatArray[] expertInputs,     // Input tokens for each expert
        int[][] tokenMappings,         // Original token indices
        float[][] tokenWeights,        // Routing weights
        int[] expertTokenCounts        // Number of tokens per expert
    ) {}

    /**
     * MoE layer statistics.
     */
    public record MoELayerStats(
        long totalForwardPasses,
        long totalExpertComputations,
        float averageExpertUtilization,
        int loadedExperts,
        DeepSeekR1ExpertRouter.RoutingStats routingStats
    ) {}

    /**
     * Get memory usage of this MoE layer.
     */
    public long getMemoryUsage() {
        long usage = 0;

        // Expert memory
        for (Expert expert : experts) {
            if (expert.isLoaded()) {
                usage += expert.getMemoryUsage();
            }
        }

        // Buffer memory
        usage += combinedOutput.getSize() * 4L;
        for (FloatArray expertOutput : expertOutputs) {
            usage += expertOutput.getSize() * 4L;
        }

        return usage;
    }

    /**
     * Unload unused experts to free memory.
     */
    public void unloadUnusedExperts() {
        for (int i = 0; i < experts.length; i++) {
            // In a real implementation, this would check usage statistics
            // and unload experts that haven't been used recently
            expertLoaded[i] = false;
        }
    }
}