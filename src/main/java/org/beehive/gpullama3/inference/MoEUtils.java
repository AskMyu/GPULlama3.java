package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import java.util.Arrays;

/**
 * Utility class for Mixture-of-Experts (MoE) operations.
 *
 * Provides core functionality for expert routing, selection, and aggregation
 * used by OLMoE and other MoE models.
 */
public final class MoEUtils {

    private MoEUtils() {
        // Utility class - prevent instantiation
    }

    /**
     * Selects top-K experts based on router logits using greedy selection.
     *
     * @param routerLogits Array of logits for all experts [numExperts]
     * @param k Number of experts to select (e.g., 8 for OLMoE)
     * @return Array of selected expert indices, sorted by score descending
     */
    public static int[] selectTopKExperts(float[] routerLogits, int k) {
        if (routerLogits == null || routerLogits.length == 0) {
            throw new IllegalArgumentException("Router logits cannot be null or empty");
        }
        if (k <= 0 || k > routerLogits.length) {
            throw new IllegalArgumentException("k must be between 1 and " + routerLogits.length);
        }

        // Create array of (index, score) pairs
        ExpertScore[] expertScores = new ExpertScore[routerLogits.length];
        for (int i = 0; i < routerLogits.length; i++) {
            expertScores[i] = new ExpertScore(i, routerLogits[i]);
        }

        // Sort by score descending
        Arrays.sort(expertScores, (a, b) -> Float.compare(b.score, a.score));

        // Extract top-K expert indices
        int[] selectedExperts = new int[k];
        for (int i = 0; i < k; i++) {
            selectedExperts[i] = expertScores[i].index;
        }

        return selectedExperts;
    }

    /**
     * Computes expert weights using softmax normalization for selected experts.
     *
     * @param routerLogits Full router logits array [numExperts]
     * @param selectedExperts Array of selected expert indices [k]
     * @return Normalized weights for selected experts [k]
     */
    public static float[] computeExpertWeights(float[] routerLogits, int[] selectedExperts) {
        if (routerLogits == null || selectedExperts == null) {
            throw new IllegalArgumentException("Inputs cannot be null");
        }

        float[] expertWeights = new float[selectedExperts.length];

        // Extract logits for selected experts
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int expertId : selectedExperts) {
            maxLogit = Math.max(maxLogit, routerLogits[expertId]);
        }

        // Compute softmax (numerically stable)
        float sumExp = 0.0f;
        for (int i = 0; i < selectedExperts.length; i++) {
            float logit = routerLogits[selectedExperts[i]] - maxLogit;
            expertWeights[i] = (float) Math.exp(logit);
            sumExp += expertWeights[i];
        }

        // Normalize
        if (sumExp > 0.0f) {
            for (int i = 0; i < expertWeights.length; i++) {
                expertWeights[i] /= sumExp;
            }
        } else {
            // Fallback: uniform weights
            Arrays.fill(expertWeights, 1.0f / expertWeights.length);
        }

        return expertWeights;
    }

    /**
     * Computes expert FFN (Feed-Forward Network) output for a single expert.
     * Implements: SwiGLU(gate(x)) * up(x) -> down(result)
     *
     * @param input Input tensor [dim]
     * @param gateWeight Gate projection weight [hiddenDim, dim]
     * @param upWeight Up projection weight [hiddenDim, dim]
     * @param downWeight Down projection weight [dim, hiddenDim]
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return Expert output [dim]
     */
    public static float[] computeExpertFFN(float[] input, float[] gateWeight,
                                         float[] upWeight, float[] downWeight,
                                         int hiddenDim, int dim) {
        if (input.length != dim) {
            throw new IllegalArgumentException("Input dimension mismatch: expected " + dim + ", got " + input.length);
        }

        // Gate projection: gate = gateWeight @ input
        float[] gate = new float[hiddenDim];
        for (int h = 0; h < hiddenDim; h++) {
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                sum += gateWeight[h * dim + d] * input[d];
            }
            gate[h] = sum;
        }

        // Up projection: up = upWeight @ input
        float[] up = new float[hiddenDim];
        for (int h = 0; h < hiddenDim; h++) {
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                sum += upWeight[h * dim + d] * input[d];
            }
            up[h] = sum;
        }

        // SwiGLU activation: silu(gate) * up
        float[] intermediate = new float[hiddenDim];
        for (int h = 0; h < hiddenDim; h++) {
            // SiLU (Swish) activation: x / (1 + exp(-x))
            float gateValue = gate[h];
            float silu = gateValue / (1.0f + (float) Math.exp(-gateValue));
            intermediate[h] = silu * up[h];
        }

        // Down projection: output = downWeight @ intermediate
        float[] output = new float[dim];
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            for (int h = 0; h < hiddenDim; h++) {
                sum += downWeight[d * hiddenDim + h] * intermediate[h];
            }
            output[d] = sum;
        }

        return output;
    }

    /**
     * Aggregates expert outputs using weighted combination.
     *
     * @param expertOutputs Array of expert outputs [numSelectedExperts][dim]
     * @param expertWeights Normalized weights for experts [numSelectedExperts]
     * @param dim Model dimension size
     * @return Aggregated output [dim]
     */
    public static float[] aggregateExpertOutputs(float[][] expertOutputs, float[] expertWeights, int dim) {
        if (expertOutputs == null || expertWeights == null) {
            throw new IllegalArgumentException("Inputs cannot be null");
        }
        if (expertOutputs.length != expertWeights.length) {
            throw new IllegalArgumentException("Expert outputs and weights length mismatch");
        }

        float[] aggregatedOutput = new float[dim];

        for (int i = 0; i < expertOutputs.length; i++) {
            float weight = expertWeights[i];
            float[] output = expertOutputs[i];

            if (output.length != dim) {
                throw new IllegalArgumentException("Expert output dimension mismatch at index " + i);
            }

            for (int d = 0; d < dim; d++) {
                aggregatedOutput[d] += weight * output[d];
            }
        }

        return aggregatedOutput;
    }

    /**
     * Computes auxiliary loss for load balancing.
     * Implements the standard MoE auxiliary loss to encourage uniform expert utilization.
     *
     * @param expertCounts Number of tokens routed to each expert [numExperts]
     * @param totalTokens Total number of tokens processed
     * @param coefficient Auxiliary loss coefficient (e.g., 0.01 for OLMoE)
     * @return Auxiliary loss value
     */
    public static float computeAuxiliaryLoss(int[] expertCounts, int totalTokens, float coefficient) {
        if (expertCounts == null || expertCounts.length == 0) {
            return 0.0f;
        }
        if (totalTokens <= 0) {
            return 0.0f;
        }

        int numExperts = expertCounts.length;
        float idealLoad = (float) totalTokens / numExperts;
        float imbalance = 0.0f;

        for (int count : expertCounts) {
            imbalance += Math.abs(count - idealLoad);
        }

        return coefficient * (imbalance / totalTokens);
    }

    /**
     * Computes router logits given input and router weights.
     *
     * CRITICAL FIX: Router weights in GGUF are stored as [dim, num_experts] not [num_experts, dim]
     * This means for each input dimension i, the weights for all experts are contiguous.
     *
     * @param input Input tensor [dim]
     * @param routerWeight Router weight matrix [dim, numExperts] - FIXED: was incorrectly documented as [numExperts, dim]
     * @param numExperts Number of experts
     * @param dim Model dimension
     * @return Router logits [numExperts]
     */
    public static float[] computeRouterLogits(float[] input, float[] routerWeight,
                                            int numExperts, int dim) {
        if (input.length != dim) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        if (routerWeight.length != numExperts * dim) {
            throw new IllegalArgumentException("Router weight dimension mismatch");
        }

        float[] routerLogits = new float[numExperts];

        // FIXED: Changed indexing to match actual GGUF storage format [dim, numExperts]
        // Compute: routerLogits[expert] = sum(input[i] * routerWeights[i][expert])
        for (int e = 0; e < numExperts; e++) {
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                // Router weights stored as [dim, numExperts]
                // So weight for input[d] -> expert[e] is at index: d * numExperts + e
                sum += input[d] * routerWeight[d * numExperts + e];
            }
            routerLogits[e] = sum;
        }

        return routerLogits;
    }

    /**
     * Helper class for expert scoring and selection.
     */
    private static class ExpertScore {
        final int index;
        final float score;

        ExpertScore(int index, float score) {
            this.index = index;
            this.score = score;
        }
    }
}