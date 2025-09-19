package org.beehive.gpullama3.model.deepseekr1;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Expert router implementation specific to DeepSeek-R1 architecture.
 *
 * Implements TopK routing with load balancing for 256 experts per layer.
 * This is completely isolated from existing MoE implementations.
 */
public class DeepSeekR1ExpertRouter {

    private final DeepSeekR1Configuration config;
    private final FloatArray routingWeights;  // [inputDim, numExperts]
    private final FloatArray routingBias;     // [numExperts]

    // Load balancing tracking
    private final FloatArray expertLoadCounts;    // Track expert usage
    private final FloatArray expertCapacities;    // Expert capacity limits
    private long totalRoutingCalls = 0;

    // Routing buffers (reused across calls)
    private final FloatArray logits;          // [batchSize * seqLen, numExperts]
    private final FloatArray probabilities;   // [batchSize * seqLen, numExperts]
    private final IntArray selectedExperts;   // [batchSize * seqLen, activeExperts]
    private final FloatArray expertWeights;   // [batchSize * seqLen, activeExperts]

    public DeepSeekR1ExpertRouter(DeepSeekR1Configuration config) {
        this.config = config;

        int inputDim = config.dim();
        int numExperts = config.totalExperts();
        int maxTokens = 1024; // Reasonable buffer size

        // Initialize routing parameters
        this.routingWeights = new FloatArray(inputDim * numExperts);
        this.routingBias = new FloatArray(numExperts);

        // Load balancing arrays
        this.expertLoadCounts = new FloatArray(numExperts);
        this.expertCapacities = new FloatArray(numExperts);

        // Routing buffers
        this.logits = new FloatArray(maxTokens * numExperts);
        this.probabilities = new FloatArray(maxTokens * numExperts);
        this.selectedExperts = new IntArray(maxTokens * config.activeExperts());
        this.expertWeights = new FloatArray(maxTokens * config.activeExperts());

        initializeRouterParameters();
    }

    /**
     * Route tokens to top-k experts with load balancing.
     *
     * @param input Input token representations [batchSize * seqLen, inputDim]
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @return Expert selection results
     */
    public ExpertSelection route(FloatArray input, int batchSize, int seqLen) {
        int numTokens = batchSize * seqLen;
        int inputDim = config.dim();
        int numExperts = config.totalExperts();
        int activeExperts = config.activeExperts();

        validateInput(input, numTokens, inputDim);

        // Step 1: Compute routing logits for all tokens
        computeRoutingLogits(input, numTokens, inputDim, numExperts);

        // Step 2: Apply softmax to get probabilities
        computeSoftmax(numTokens, numExperts);

        // Step 3: Select top-k experts for each token
        selectTopKExperts(numTokens, numExperts, activeExperts);

        // Step 4: Apply load balancing if enabled
        if (config.enableLoadBalancing()) {
            applyLoadBalancing(numTokens, activeExperts);
        }

        // Step 5: Normalize weights
        normalizeExpertWeights(numTokens, activeExperts);

        totalRoutingCalls++;

        return new ExpertSelection(
            copyArray(selectedExperts, numTokens * activeExperts),
            copyArray(expertWeights, numTokens * activeExperts),
            batchSize,
            seqLen,
            activeExperts
        );
    }

    /**
     * Compute routing logits: input * routingWeights + routingBias
     */
    private void computeRoutingLogits(FloatArray input, int numTokens, int inputDim, int numExperts) {
        for (int token = 0; token < numTokens; token++) {
            for (int expert = 0; expert < numExperts; expert++) {
                float logit = routingBias.get(expert);

                // Dot product with routing weights
                for (int dim = 0; dim < inputDim; dim++) {
                    float inputVal = input.get(token * inputDim + dim);
                    float weight = routingWeights.get(dim * numExperts + expert);
                    logit += inputVal * weight;
                }

                logits.set(token * numExperts + expert, logit);
            }
        }
    }

    /**
     * Apply softmax to routing logits to get probabilities.
     */
    private void computeSoftmax(int numTokens, int numExperts) {
        for (int token = 0; token < numTokens; token++) {
            int baseIdx = token * numExperts;

            // Find max for numerical stability
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int expert = 0; expert < numExperts; expert++) {
                maxLogit = Math.max(maxLogit, logits.get(baseIdx + expert));
            }

            // Compute exp and sum
            float sumExp = 0.0f;
            for (int expert = 0; expert < numExperts; expert++) {
                float expLogit = (float) Math.exp(logits.get(baseIdx + expert) - maxLogit);
                probabilities.set(baseIdx + expert, expLogit);
                sumExp += expLogit;
            }

            // Normalize
            for (int expert = 0; expert < numExperts; expert++) {
                float prob = probabilities.get(baseIdx + expert) / sumExp;
                probabilities.set(baseIdx + expert, prob);
            }
        }
    }

    /**
     * Select top-k experts for each token based on routing probabilities.
     */
    private void selectTopKExperts(int numTokens, int numExperts, int activeExperts) {
        for (int token = 0; token < numTokens; token++) {
            int probBaseIdx = token * numExperts;
            int selectionBaseIdx = token * activeExperts;

            // Create array of (probability, expertIndex) pairs
            float[] probs = new float[numExperts];
            int[] indices = new int[numExperts];

            for (int expert = 0; expert < numExperts; expert++) {
                probs[expert] = probabilities.get(probBaseIdx + expert);
                indices[expert] = expert;
            }

            // Partial sort to find top-k (using simple selection for clarity)
            for (int k = 0; k < activeExperts; k++) {
                // Find max remaining
                int maxIdx = k;
                for (int i = k + 1; i < numExperts; i++) {
                    if (probs[i] > probs[maxIdx]) {
                        maxIdx = i;
                    }
                }

                // Swap to position k
                if (maxIdx != k) {
                    float tempProb = probs[k];
                    int tempIdx = indices[k];
                    probs[k] = probs[maxIdx];
                    indices[k] = indices[maxIdx];
                    probs[maxIdx] = tempProb;
                    indices[maxIdx] = tempIdx;
                }

                // Store selected expert and weight
                selectedExperts.set(selectionBaseIdx + k, indices[k]);
                expertWeights.set(selectionBaseIdx + k, probs[k]);
            }
        }
    }

    /**
     * Apply load balancing to distribute load more evenly across experts.
     */
    private void applyLoadBalancing(int numTokens, int activeExperts) {
        // Update expert load counts
        for (int token = 0; token < numTokens; token++) {
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selectedExperts.get(token * activeExperts + k);
                float currentLoad = expertLoadCounts.get(expertIdx);
                expertLoadCounts.set(expertIdx, currentLoad + 1.0f);
            }
        }

        // Apply load balancing penalty
        float avgLoad = (float) totalRoutingCalls / config.totalExperts();
        float balancingStrength = 0.01f; // Hyperparameter

        for (int token = 0; token < numTokens; token++) {
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selectedExperts.get(token * activeExperts + k);
                float expertLoad = expertLoadCounts.get(expertIdx);
                float loadPenalty = balancingStrength * Math.max(0, expertLoad - avgLoad);

                float currentWeight = expertWeights.get(token * activeExperts + k);
                expertWeights.set(token * activeExperts + k, currentWeight * (1.0f - loadPenalty));
            }
        }
    }

    /**
     * Normalize expert weights so they sum to 1.0 for each token.
     */
    private void normalizeExpertWeights(int numTokens, int activeExperts) {
        for (int token = 0; token < numTokens; token++) {
            int baseIdx = token * activeExperts;

            // Compute sum of weights for this token
            float sumWeights = 0.0f;
            for (int k = 0; k < activeExperts; k++) {
                sumWeights += expertWeights.get(baseIdx + k);
            }

            // Normalize weights
            if (sumWeights > 0.0f) {
                for (int k = 0; k < activeExperts; k++) {
                    float weight = expertWeights.get(baseIdx + k) / sumWeights;
                    expertWeights.set(baseIdx + k, weight);
                }
            }
        }
    }

    /**
     * Set routing parameters from trained model weights.
     */
    public void setRoutingWeights(FloatArray weights, FloatArray bias) {
        if (weights.getSize() != routingWeights.getSize()) {
            throw new IllegalArgumentException("Routing weights size mismatch");
        }
        if (bias.getSize() != routingBias.getSize()) {
            throw new IllegalArgumentException("Routing bias size mismatch");
        }

        // Copy weights
        for (int i = 0; i < weights.getSize(); i++) {
            routingWeights.set(i, weights.get(i));
        }

        // Copy bias
        for (int i = 0; i < bias.getSize(); i++) {
            routingBias.set(i, bias.get(i));
        }
    }

    /**
     * Initialize routing parameters with small random weights.
     */
    private void initializeRouterParameters() {
        // Xavier initialization for routing weights
        float scale = (float) Math.sqrt(1.0 / config.dim());

        for (int i = 0; i < routingWeights.getSize(); i++) {
            float value = (float) (Math.random() * 2 * scale - scale);
            routingWeights.set(i, value);
        }

        // Initialize bias to zero
        for (int i = 0; i < routingBias.getSize(); i++) {
            routingBias.set(i, 0.0f);
        }

        // Initialize load tracking
        for (int i = 0; i < expertLoadCounts.getSize(); i++) {
            expertLoadCounts.set(i, 0.0f);
            expertCapacities.set(i, Float.MAX_VALUE); // No capacity limits by default
        }
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
     * Copy array contents (utility method).
     */
    private IntArray copyArray(IntArray source, int length) {
        IntArray copy = new IntArray(length);
        for (int i = 0; i < length; i++) {
            copy.set(i, source.get(i));
        }
        return copy;
    }

    /**
     * Copy array contents (utility method).
     */
    private FloatArray copyArray(FloatArray source, int length) {
        FloatArray copy = new FloatArray(length);
        for (int i = 0; i < length; i++) {
            copy.set(i, source.get(i));
        }
        return copy;
    }

    /**
     * Get routing statistics.
     */
    public RoutingStats getStats() {
        float avgLoad = (float) totalRoutingCalls / config.totalExperts();
        float maxLoad = 0.0f;
        float minLoad = Float.MAX_VALUE;

        for (int i = 0; i < expertLoadCounts.getSize(); i++) {
            float load = expertLoadCounts.get(i);
            maxLoad = Math.max(maxLoad, load);
            minLoad = Math.min(minLoad, load);
        }

        float loadImbalance = (maxLoad - minLoad) / (avgLoad + 1e-8f);

        return new RoutingStats(totalRoutingCalls, avgLoad, maxLoad, minLoad, loadImbalance);
    }

    /**
     * Expert selection result.
     */
    public record ExpertSelection(
        IntArray selectedExperts,    // [numTokens, activeExperts]
        FloatArray expertWeights,    // [numTokens, activeExperts]
        int batchSize,
        int seqLen,
        int activeExperts
    ) {}

    /**
     * Routing statistics for monitoring.
     */
    public record RoutingStats(
        long totalRoutingCalls,
        float avgExpertLoad,
        float maxExpertLoad,
        float minExpertLoad,
        float loadImbalance
    ) {}

    /**
     * Reset load balancing statistics.
     */
    public void resetLoadBalancing() {
        for (int i = 0; i < expertLoadCounts.getSize(); i++) {
            expertLoadCounts.set(i, 0.0f);
        }
        totalRoutingCalls = 0;
    }
}