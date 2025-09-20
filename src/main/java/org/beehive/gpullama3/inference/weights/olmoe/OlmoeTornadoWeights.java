package org.beehive.gpullama3.inference.weights.olmoe;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * OLMoE-specific weights implementation for TornadoVM (GPU) computation.
 *
 * Handles the unique weight structure of Mixture-of-Experts models with
 * separate router and expert tensors optimized for GPU execution.
 *
 * Since OLMoE uses expert weights instead of standard FFN w1/w2/w3 weights,
 * this class extends TornadoWeights but provides expert-specific functionality.
 */
public class OlmoeTornadoWeights extends TornadoWeights {

    // MoE-specific weights (additional to standard TornadoWeights)
    private final FloatArray[] routerWeights;       // [layers] -> [dim, num_experts]
    private final FloatArray[] expertGateWeights;   // [layers] -> [num_experts * hidden_dim, dim]
    private final FloatArray[] expertDownWeights;   // [layers] -> [num_experts * dim, hidden_dim]
    private final FloatArray[] expertUpWeights;     // [layers] -> [num_experts * hidden_dim, dim]

    public OlmoeTornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqLayered,
            HalfFloatArray[] wkLayered,
            HalfFloatArray[] wvLayered,
            HalfFloatArray[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            HalfFloatArray[] w1Layered,  // These will be null/placeholders for OLMoE
            HalfFloatArray[] w2Layered,  // These will be null/placeholders for OLMoE
            HalfFloatArray[] w3Layered,  // These will be null/placeholders for OLMoE
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            HalfFloatArray wclsHalfFloat,
            GGMLType weightType,
            FloatArray[] routerWeights,
            FloatArray[] expertGateWeights,
            FloatArray[] expertDownWeights,
            FloatArray[] expertUpWeights) {

        // Call parent TornadoWeights constructor
        super(tokenEmbeddingTable,
              rms_att_weightLayered,
              wqLayered,
              wkLayered,
              wvLayered,
              woLayered,
              rms_ffn_weightLayered,
              w1Layered,  // For OLMoE, these are placeholders since we use expert weights
              w2Layered,
              w3Layered,
              rms_final_weight_as_floatArray,
              freq_cis_realFlat,
              freq_cis_imagFlat,
              wclsHalfFloat,
              weightType);

        // Store OLMoE-specific weights
        this.routerWeights = routerWeights;
        this.expertGateWeights = expertGateWeights;
        this.expertDownWeights = expertDownWeights;
        this.expertUpWeights = expertUpWeights;
    }

    // Standard weight accessors from Weights interface - delegate to inherited TornadoWeights fields
    public FloatTensor tokenEmbeddings() {
        throw new UnsupportedOperationException("Use tokenEmbeddingTable for TornadoVM weights");
    }

    public FloatTensor attentionNorm(int layer) {
        throw new UnsupportedOperationException("Use rms_att_weightLayered[layer] for TornadoVM weights");
    }

    public FloatTensor queryWeights(int layer) {
        throw new UnsupportedOperationException("Use wqLayered[layer] for TornadoVM weights");
    }

    public FloatTensor keyWeights(int layer) {
        throw new UnsupportedOperationException("Use wkLayered[layer] for TornadoVM weights");
    }

    public FloatTensor valueWeights(int layer) {
        throw new UnsupportedOperationException("Use wvLayered[layer] for TornadoVM weights");
    }

    public FloatTensor outputWeights(int layer) {
        throw new UnsupportedOperationException("Use woLayered[layer] for TornadoVM weights");
    }

    public FloatTensor ffnNorm(int layer) {
        throw new UnsupportedOperationException("Use rms_ffn_weightLayered[layer] for TornadoVM weights");
    }

    // FFN weight accessors - NOT USED for OLMoE, use expert weights instead
    public FloatTensor w1(int layer) {
        throw new UnsupportedOperationException("OLMoE uses expert weights instead of w1. Use getExpertGateWeightsSlice()");
    }

    public FloatTensor w2(int layer) {
        throw new UnsupportedOperationException("OLMoE uses expert weights instead of w2. Use getExpertDownWeightsSlice()");
    }

    public FloatTensor w3(int layer) {
        throw new UnsupportedOperationException("OLMoE uses expert weights instead of w3. Use getExpertUpWeightsSlice()");
    }

    public FloatTensor outputNorm() {
        throw new UnsupportedOperationException("Use rms_final_weight_as_floatArray for TornadoVM weights");
    }

    public FloatTensor ropeFreqsCis() {
        throw new UnsupportedOperationException("Use freq_cis_realFlat for TornadoVM weights");
    }

    public FloatTensor ropeFreqsSin() {
        throw new UnsupportedOperationException("Use freq_cis_imagFlat for TornadoVM weights");
    }

    public FloatTensor outputWeight() {
        throw new UnsupportedOperationException("Use wclsHalfFloat for TornadoVM weights");
    }

    public GGMLType outputWeightType() {
        return getWeightType();
    }

    // MoE-specific TornadoVM accessors
    public FloatArray routerWeightsArray(int layer) {
        return routerWeights[layer];
    }

    public FloatArray expertGateWeightsArray(int layer) {
        return expertGateWeights[layer];
    }

    public FloatArray expertDownWeightsArray(int layer) {
        return expertDownWeights[layer];
    }

    public FloatArray expertUpWeightsArray(int layer) {
        return expertUpWeights[layer];
    }

    /**
     * Creates a FloatArray slice for a specific expert's gate weights.
     * This is a view into the larger expert tensor, optimized for GPU access.
     *
     * @param layer Layer index
     * @param expertId Expert index
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return FloatArray slice for the expert's gate weights
     */
    public FloatArray getExpertGateWeightsSlice(int layer, int expertId, int hiddenDim, int dim) {
        FloatArray allWeights = expertGateWeights[layer];
        int expertOffset = expertId * hiddenDim * dim;
        int expertSize = hiddenDim * dim;

        // Create a slice view (if supported) or copy
        FloatArray expertSlice = new FloatArray(expertSize);
        for (int i = 0; i < expertSize; i++) {
            expertSlice.set(i, allWeights.get(expertOffset + i));
        }

        return expertSlice;
    }

    /**
     * Creates a FloatArray slice for a specific expert's down weights.
     *
     * @param layer Layer index
     * @param expertId Expert index
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return FloatArray slice for the expert's down weights
     */
    public FloatArray getExpertDownWeightsSlice(int layer, int expertId, int hiddenDim, int dim) {
        FloatArray allWeights = expertDownWeights[layer];
        int expertOffset = expertId * dim * hiddenDim;
        int expertSize = dim * hiddenDim;

        FloatArray expertSlice = new FloatArray(expertSize);
        for (int i = 0; i < expertSize; i++) {
            expertSlice.set(i, allWeights.get(expertOffset + i));
        }

        return expertSlice;
    }

    /**
     * Creates a FloatArray slice for a specific expert's up weights.
     *
     * @param layer Layer index
     * @param expertId Expert index
     * @param hiddenDim Hidden dimension size
     * @param dim Model dimension size
     * @return FloatArray slice for the expert's up weights
     */
    public FloatArray getExpertUpWeightsSlice(int layer, int expertId, int hiddenDim, int dim) {
        FloatArray allWeights = expertUpWeights[layer];
        int expertOffset = expertId * hiddenDim * dim;
        int expertSize = hiddenDim * dim;

        FloatArray expertSlice = new FloatArray(expertSize);
        for (int i = 0; i < expertSize; i++) {
            expertSlice.set(i, allWeights.get(expertOffset + i));
        }

        return expertSlice;
    }
}