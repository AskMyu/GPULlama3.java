package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.weights.Weights;

public class Qwen2StandardWeights extends StandardWeights {
    public final FloatTensor[] q_bias, k_bias, v_bias;

    public Qwen2StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] q_bias,
            FloatTensor[] k_bias,
            FloatTensor[] v_bias,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor rms_final_weight,
            ArrayFloatTensor freq_cis_real,
            ArrayFloatTensor freq_cis_imag,
            FloatTensor wcls,
            GGMLType weightType) {
        // call to StandardWeights constructor
        super(token_embedding_table,
                rms_att_weight,
                wq,
                wk,
                wv,
                wo,
                rms_ffn_weight,
                w1,
                w2,
                w3,
                rms_final_weight,
                freq_cis_real,
                freq_cis_imag,
                wcls,
                weightType);
        // init Qwen2-specific fields
        this.q_bias = q_bias;
        this.k_bias = k_bias;
        this.v_bias = v_bias;
    }

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
