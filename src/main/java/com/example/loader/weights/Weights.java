package com.example.loader.weights;

import com.example.LlamaApp;
import com.example.core.model.GGMLType;
import com.example.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.FloatBuffer;

public class Weights {
    // token embedding table
    public final FloatTensor token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls
    public final FloatTensor[] wq; // (layer, n_heads * head_size)
    public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
    public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
    public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
    public final FloatBuffer[] rms_ffn_weight; // (layer, dim)

    // weights for ffn
    public final FloatTensor[] w1; // (layer, hidden_dim, dim)
    public final FloatTensor[] w2; // (layer, dim, hidden_dim)
    public final FloatTensor[] w3; // (layer, hidden_dim, dim)
    //
    public final FloatTensor wcls; // (vocab_size, dim)
    public final ByteArray wclsByteArray;
    // public final rmsnorm
    public final FloatBuffer rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
    public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
    //    // Layered Data structures
    public FloatArray[] rms_att_weightLayered; // (layer, dim) rmsnorm weights
    public HalfFloatArray[] wqLayered; // (layer, n_heads * head_size)
    public HalfFloatArray[] wkLayered; // (layer, n_kv_heads, head_size)
    public HalfFloatArray[] wvLayered; // (layer, n_kv_heads * head_size)
    public HalfFloatArray[] woLayered; // (layer, n_heads * head_size, dim)
    public FloatArray[] rms_ffn_weightLayered; // (layer, dim)
    public HalfFloatArray[] w1Layered; // (layer, hidden_dim, dim)
    public HalfFloatArray[] w2Layered; // (layer, dim, hidden_dim)
    //
    public HalfFloatArray[] w3Layered; // (layer, hidden_dim, dim)
    public FloatArray rms_final_weight_as_floatArray;
    public FloatArray tokenEmbeddingTable; // (vocab_size, dim)
    public FloatArray freq_cis_realFlat; // (seq_len, head_size/2)
    public FloatArray freq_cis_imagFlat; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    public GGMLType weightType;
    /**
     * Constructor to initialize all weight tensors for the model. Automatically creates TornadoVM-compatible versions when needed.
     *
     * @param token_embedding_table
     *         Token embeddings matrix
     * @param rms_att_weight
     *         RMSNorm weights for attention layers
     * @param wq
     *         Query weight matrices
     * @param wk
     *         Key weight matrices
     * @param wv
     *         Value weight matrices
     * @param wo
     *         Output projection matrices
     * @param rms_ffn_weight
     *         RMSNorm weights for FFN layers
     * @param w1
     *         First FFN weight matrices
     * @param w2
     *         Second FFN weight matrices
     * @param w3
     *         Third FFN weight matrices (gate)
     * @param rms_final_weight
     *         Final layer normalization weights
     * @param freq_cis_real
     *         RoPE cosine components
     * @param freq_cis_imag
     *         RoPE sine components
     * @param wcls
     *         Classifier weights for output logits
     *
    /**
     * Constructor for standard (non-TornadoVM) mode
     */
    public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight,
            FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
            FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3,
            FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag,
            FloatTensor wcls, GGMLType weightType) {
        // Standard format
        this.token_embedding_table = token_embedding_table;
        this.rms_att_weight = rms_att_weight;
        this.wq = wq; this.wk = wk; this.wv = wv; this.wo = wo;
        this.rms_ffn_weight = rms_ffn_weight;
        this.w1 = w1; this.w2 = w2; this.w3 = w3;
        this.wcls = wcls;
        this.rms_final_weight = rms_final_weight;
        this.freq_cis_real = freq_cis_real;
        this.freq_cis_imag = freq_cis_imag;
        this.weightType = weightType;

        // TornadoVM format (null when not using TornadoVM)
        this.tokenEmbeddingTable = null;
        this.rms_att_weightLayered = null;
        this.wqLayered = null; this.wkLayered = null; this.wvLayered = null; this.woLayered = null;
        this.rms_ffn_weightLayered = null;
        this.w1Layered = null; this.w2Layered = null; this.w3Layered = null;
        this.rms_final_weight_as_floatArray = null;
        this.freq_cis_realFlat = null; this.freq_cis_imagFlat = null;
        this.wclsByteArray = null;
    }

    /**
     * Constructor for TornadoVM mode
     */
    public Weights(FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqLayered, HalfFloatArray[] wkLayered, HalfFloatArray[] wvLayered, HalfFloatArray[] woLayered,
            FloatArray[] rms_ffn_weightLayered, HalfFloatArray[] w1Layered, HalfFloatArray[] w2Layered, HalfFloatArray[] w3Layered,
            FloatArray rms_final_weight_as_floatArray, FloatArray freq_cis_realFlat, FloatArray freq_cis_imagFlat,
            ByteArray wclsByteArray, GGMLType weightType) {
        // Standard format (null when using TornadoVM)
        this.token_embedding_table = null;
        this.rms_att_weight = null;
        this.wq = null; this.wk = null; this.wv = null; this.wo = null;
        this.rms_ffn_weight = null;
        this.w1 = null; this.w2 = null; this.w3 = null;
        this.wcls = null;
        this.rms_final_weight = null;
        this.freq_cis_real = null; this.freq_cis_imag = null;

        // TornadoVM format
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.rms_att_weightLayered = rms_att_weightLayered;
        this.wqLayered = wqLayered; this.wkLayered = wkLayered; this.wvLayered = wvLayered; this.woLayered = woLayered;
        this.rms_ffn_weightLayered = rms_ffn_weightLayered;
        this.w1Layered = w1Layered; this.w2Layered = w2Layered; this.w3Layered = w3Layered;
        this.rms_final_weight_as_floatArray = rms_final_weight_as_floatArray;
        this.freq_cis_realFlat = freq_cis_realFlat; this.freq_cis_imagFlat = freq_cis_imagFlat;
        this.wclsByteArray = wclsByteArray;
        this.weightType = weightType;
    }

    /**
     * Converts an array of FloatBuffer objects to TornadoVM FloatArray format. Preserves the original buffer position after conversion.
     *
     * @param array
     *         Array of FloatBuffers to convert
     * @return Array of FloatArrays with the same data
     */
    private static FloatArray[] loadToFloatArray(FloatBuffer[] array) {
        FloatArray[] result = new FloatArray[array.length];
        for (int i = 0; i < array.length; i++) {
            int size = array[i].remaining();
            result[i] = new FloatArray(size);

            // Save and restore buffer position to avoid side effects
            int originalPosition = array[i].position();

            for (int j = 0; j < size; j++) {
                float value = array[i].get();
                result[i].set(j, value);
            }
            // Reset buffer position
            array[i].position(originalPosition);
        }

        return result;
    }


    /**
     * Converts a single FloatBuffer to a TornadoVM FloatArray. Creates a duplicate buffer to avoid modifying the original.
     *
     * @param input
     *         FloatBuffer to convert
     * @return FloatArray with the same data
     */
    private static FloatArray loadToSingleFloatArray(FloatBuffer input) {
        // Create a duplicate to prevent modifying the original buffer
        FloatBuffer copy = input.duplicate();
        int totalSize = copy.remaining();

        FloatArray result = new FloatArray(totalSize);

        int index = 0;
        while (copy.hasRemaining()) {
            result.set(index++, copy.get());
        }

        return result;
    }

    /**
     * Converts a FloatTensor to a TornadoVM FloatArray.
     *
     * @param input
     *         FloatTensor to convert
     * @return FloatArray with the same data
     */
    public FloatArray loadToFloatArray(FloatTensor input) {
        FloatArray floatArray = new FloatArray(input.size());

        for (int i = 0; i < input.size(); i++) {
            floatArray.set(i, input.getFloat(i));
        }

        return floatArray;
    }

}