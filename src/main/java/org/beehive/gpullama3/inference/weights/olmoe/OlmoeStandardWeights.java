package org.beehive.gpullama3.inference.weights.olmoe;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.weights.Weights;

/**
 * OLMoE-specific weights implementation for standard (CPU) computation.
 *
 * Handles the unique weight structure of Mixture-of-Experts models with
 * separate router and expert tensors instead of standard FFN weights.
 */
public class OlmoeStandardWeights implements Weights {

    // Standard transformer weights (same as Llama)
    private final FloatTensor tokenEmbeddings;
    private final FloatTensor[] attentionNorms;      // [layers]
    private final FloatTensor[] queryWeights;        // [layers]
    private final FloatTensor[] keyWeights;          // [layers]
    private final FloatTensor[] valueWeights;        // [layers]
    private final FloatTensor[] outputWeights;       // [layers]
    private final FloatTensor[] ffnNorms;            // [layers]

    // MoE-specific weights
    private final FloatTensor[] routerWeights;       // [layers] -> [dim, num_experts]
    private final FloatTensor[] expertGateWeights;   // [layers] -> [num_experts * hidden_dim, dim]
    private final FloatTensor[] expertDownWeights;   // [layers] -> [num_experts * dim, hidden_dim]
    private final FloatTensor[] expertUpWeights;     // [layers] -> [num_experts * hidden_dim, dim]

    // Output and RoPE weights (same as Llama)
    private final FloatTensor outputNorm;
    private final FloatTensor ropeFreqsCis;
    private final FloatTensor ropeFreqsSin;
    private final FloatTensor outputWeight;
    private final GGMLType outputWeightType;

    public OlmoeStandardWeights(
            FloatTensor tokenEmbeddings,
            FloatTensor[] attentionNorms,
            FloatTensor[] queryWeights,
            FloatTensor[] keyWeights,
            FloatTensor[] valueWeights,
            FloatTensor[] outputWeights,
            FloatTensor[] ffnNorms,
            FloatTensor[] routerWeights,
            FloatTensor[] expertGateWeights,
            FloatTensor[] expertDownWeights,
            FloatTensor[] expertUpWeights,
            FloatTensor outputNorm,
            FloatTensor ropeFreqsCis,
            FloatTensor ropeFreqsSin,
            FloatTensor outputWeight,
            GGMLType outputWeightType) {

        this.tokenEmbeddings = tokenEmbeddings;
        this.attentionNorms = attentionNorms;
        this.queryWeights = queryWeights;
        this.keyWeights = keyWeights;
        this.valueWeights = valueWeights;
        this.outputWeights = outputWeights;
        this.ffnNorms = ffnNorms;
        this.routerWeights = routerWeights;
        this.expertGateWeights = expertGateWeights;
        this.expertDownWeights = expertDownWeights;
        this.expertUpWeights = expertUpWeights;
        this.outputNorm = outputNorm;
        this.ropeFreqsCis = ropeFreqsCis;
        this.ropeFreqsSin = ropeFreqsSin;
        this.outputWeight = outputWeight;
        this.outputWeightType = outputWeightType;
    }

    // Standard weight accessors (same as Llama)
    public FloatTensor tokenEmbeddings() {
        return tokenEmbeddings;
    }

    public FloatTensor attentionNorm(int layer) {
        return attentionNorms[layer];
    }

    public FloatTensor queryWeights(int layer) {
        return queryWeights[layer];
    }

    public FloatTensor keyWeights(int layer) {
        return keyWeights[layer];
    }

    public FloatTensor valueWeights(int layer) {
        return valueWeights[layer];
    }

    public FloatTensor outputWeights(int layer) {
        return outputWeights[layer];
    }

    public FloatTensor ffnNorm(int layer) {
        return ffnNorms[layer];
    }

    // FFN weight accessors - NOT USED for OLMoE, use expert weights instead
    public FloatTensor w1(int layer) {
        throw new UnsupportedOperationException("OLMoE uses expert weights instead of w1. Use expertGateWeights()");
    }

    public FloatTensor w2(int layer) {
        throw new UnsupportedOperationException("OLMoE uses expert weights instead of w2. Use expertDownWeights()");
    }

    public FloatTensor w3(int layer) {
        throw new UnsupportedOperationException("OLMoE uses expert weights instead of w3. Use expertUpWeights()");
    }

    // MoE-specific weight accessors
    public FloatTensor routerWeights(int layer) {
        return routerWeights[layer];
    }

    public FloatTensor expertGateWeights(int layer) {
        return expertGateWeights[layer];
    }

    public FloatTensor expertDownWeights(int layer) {
        return expertDownWeights[layer];
    }

    public FloatTensor expertUpWeights(int layer) {
        return expertUpWeights[layer];
    }

    /**
     * Extracts gate weights for a specific expert.
     *
     * @param layer Layer index
     * @param expertId Expert index (0 to numExperts-1)
     * @param numExperts Total number of experts
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return Gate weights for the specified expert [hiddenDim, dim]
     */
    public float[] getExpertGateWeights(int layer, int expertId, int numExperts, int hiddenDim, int dim) {
        FloatTensor allGateWeights = expertGateWeights[layer];
        float[] expertWeights = new float[hiddenDim * dim];

        int expertOffset = expertId * hiddenDim * dim;
        for (int i = 0; i < hiddenDim * dim; i++) {
            expertWeights[i] = allGateWeights.getFloat(expertOffset + i);
        }

        return expertWeights;
    }

    /**
     * Extracts down weights for a specific expert.
     *
     * @param layer Layer index
     * @param expertId Expert index (0 to numExperts-1)
     * @param numExperts Total number of experts
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return Down weights for the specified expert [dim, hiddenDim]
     */
    public float[] getExpertDownWeights(int layer, int expertId, int numExperts, int hiddenDim, int dim) {
        FloatTensor allDownWeights = expertDownWeights[layer];
        float[] expertWeights = new float[dim * hiddenDim];

        int expertOffset = expertId * dim * hiddenDim;
        for (int i = 0; i < dim * hiddenDim; i++) {
            expertWeights[i] = allDownWeights.getFloat(expertOffset + i);
        }

        return expertWeights;
    }

    /**
     * Extracts up weights for a specific expert.
     *
     * @param layer Layer index
     * @param expertId Expert index (0 to numExperts-1)
     * @param numExperts Total number of experts
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return Up weights for the specified expert [hiddenDim, dim]
     */
    public float[] getExpertUpWeights(int layer, int expertId, int numExperts, int hiddenDim, int dim) {
        FloatTensor allUpWeights = expertUpWeights[layer];
        float[] expertWeights = new float[hiddenDim * dim];

        int expertOffset = expertId * hiddenDim * dim;
        for (int i = 0; i < hiddenDim * dim; i++) {
            expertWeights[i] = allUpWeights.getFloat(expertOffset + i);
        }

        return expertWeights;
    }

    /**
     * Extracts router weights for computing expert logits.
     *
     * @param layer Layer index
     * @return Router weights [numExperts, dim]
     */
    public float[] getRouterWeights(int layer) {
        FloatTensor routerTensor = routerWeights[layer];
        int size = routerTensor.size();
        float[] weights = new float[size];

        for (int i = 0; i < size; i++) {
            weights[i] = routerTensor.getFloat(i);
        }

        return weights;
    }

    // Output and RoPE accessors (same as Llama)
    public FloatTensor outputNorm() {
        return outputNorm;
    }

    public FloatTensor ropeFreqsCis() {
        return ropeFreqsCis;
    }

    public FloatTensor ropeFreqsSin() {
        return ropeFreqsSin;
    }

    public FloatTensor outputWeight() {
        return outputWeight;
    }

    public GGMLType outputWeightType() {
        return outputWeightType;
    }

    public GGMLType getWeightType() {
        return outputWeightType;
    }
}