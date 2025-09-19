package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.standard.StandardWeights;
import org.beehive.gpullama3.model.granite.Granite;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;

/**
 * Granite-specific inference core with GQA (Group-Query Attention) and SwiGLU support.
 *
 * This implementation fixes the NaN issues by properly implementing:
 * 1. Group-Query Attention where KV heads are shared across Q heads
 * 2. SwiGLU activation function instead of standard GELU
 * 3. Proper dimension handling for Granite's architecture
 */
public class GraniteInferenceCore {

    private static final boolean DEBUG = true; // Force debug output
    private static final boolean VALIDATE_NAN = System.getProperty("granite.validate.nan", "true").equals("true");

    /**
     * Main forward pass for Granite models.
     * Uses Group-Query Attention and SwiGLU activation.
     */
    public static void forwardGranite(Granite model, State state, int token, int position) {
        System.err.println("🔥 GRANITE-INFERENCE: forwardGranite() called!"); // Force output

        GraniteConfiguration config = model.configuration();
        StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int kvDim = config.kvDim();
        int nLayers = config.numberOfLayers();

        // ALWAYS log configuration - no DEBUG check
        System.err.printf("🔥 GRANITE-CONFIG: dim=%d, kvDim=%d, nHeads=%d, nKVHeads=%d, headSize=%d, kvMul=%d\n",
                        dim, kvDim, config.numberOfHeads(), config.numberOfKeyValueHeads(),
                        config.headSize(), config.kvMul());

        if (DEBUG) {
            System.err.printf("[GRANITE-FORWARD] Starting forward pass: token=%d, position=%d\n", token, position);
            System.err.printf("[GRANITE-CONFIG] dim=%d, kvDim=%d, nHeads=%d, nKVHeads=%d, headSize=%d, kvMul=%d\n",
                            dim, kvDim, config.numberOfHeads(), config.numberOfKeyValueHeads(),
                            config.headSize(), config.kvMul());
            System.err.printf("[GRANITE-WEIGHTS] token_embedding_table size=%d, expected=%d\n",
                            weights.token_embedding_table.size(), config.vocabularySize() * dim);

            // Validate weight matrix dimensions
            if (weights.wq != null && weights.wq.length > 0) {
                System.err.printf("[GRANITE-WEIGHTS] wq[0] size=%d, expected=%d\n",
                                weights.wq[0].size(), dim * dim);
            }
            if (weights.wk != null && weights.wk.length > 0) {
                System.err.printf("[GRANITE-WEIGHTS] wk[0] size=%d, expected=%d\n",
                                weights.wk[0].size(), kvDim * dim);
            }
            if (weights.wv != null && weights.wv.length > 0) {
                System.err.printf("[GRANITE-WEIGHTS] wv[0] size=%d, expected=%d\n",
                                weights.wv[0].size(), kvDim * dim);
            }
        }

        // 1. Token Embedding
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        if (VALIDATE_NAN) {
            validateNoNaN(state.x, "initial embedding");
        }

        // 2. Process through layers with GQA and SwiGLU
        for (int layer = 0; layer < nLayers; layer++) {
            if (DEBUG && layer < 3) {
                System.err.printf("[GRANITE-LAYER] Processing layer %d/%d\n", layer, nLayers);
            }

            // Save input for residual connection
            for (int i = 0; i < state.xb.size(); i++) {
                state.xb.setFloat(i, state.x.getFloat(i));
            }

            // RMSNorm before attention
            InferenceCore.rmsnorm(state.xb2, state.x, weights.rms_att_weight[layer],
                                 0, dim, config.rmsNormEps());

            // GROUP-QUERY ATTENTION (GQA)
            computeGQA(state, weights, layer, position, config);

            // Residual connection after attention
            state.x.addInPlace(state.xb2);

            // Save for next residual
            for (int i = 0; i < state.xb.size(); i++) {
                state.xb.setFloat(i, state.x.getFloat(i));
            }

            // RMSNorm before FFN
            InferenceCore.rmsnorm(state.xb2, state.x, weights.rms_ffn_weight[layer],
                                 0, dim, config.rmsNormEps());

            // SwiGLU activation (Granite's key difference)
            computeSwiGLU(state, weights, layer, config);

            // Final residual connection
            state.x.addInPlace(state.xb2);

            if (VALIDATE_NAN) {
                validateNoNaN(state.x, "after layer " + layer);
            }

            if (DEBUG && layer < 3) {
                float norm = computeNorm(state.x, dim);
                System.err.printf("[GRANITE-LAYER] Layer %d complete: norm=%.6f\n", layer, norm);
            }
        }

        // 3. Final RMSNorm and output projection
        InferenceCore.rmsnorm(state.x, state.x, weights.rms_final_weight,
                             0, dim, config.rmsNormEps());

        // Compute logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        if (VALIDATE_NAN) {
            validateNoNaN(state.logits, "final logits");
        }

        if (DEBUG) {
            System.err.printf("[GRANITE-FORWARD] Forward pass complete, logits computed\n");

            // Debug logits values
            float maxLogit = Float.NEGATIVE_INFINITY;
            float minLogit = Float.POSITIVE_INFINITY;
            int maxToken = -1;
            int nonZeroCount = 0;

            for (int i = 0; i < Math.min(config.vocabularySize(), 1000); i++) {
                float logit = state.logits.getFloat(i);
                if (logit > maxLogit) {
                    maxLogit = logit;
                    maxToken = i;
                }
                if (logit < minLogit) {
                    minLogit = logit;
                }
                if (Math.abs(logit) > 1e-6) {
                    nonZeroCount++;
                }
            }

            System.err.printf("[GRANITE-LOGITS] Max: %.6f (token %d), Min: %.6f, NonZero: %d/1000\n",
                            maxLogit, maxToken, minLogit, nonZeroCount);
        }
    }

    /**
     * Group-Query Attention implementation.
     * Key difference from standard MHA: KV heads are shared across Q heads.
     * For Granite 2B: 16 Q heads share 4 KV heads (4:1 ratio).
     */
    private static void computeGQA(State state, StandardWeights weights, int layer,
                                   int position, GraniteConfiguration config) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int nHeads = config.numberOfHeads();
        int nKVHeads = config.numberOfKeyValueHeads();
        int kvMul = config.kvMul(); // How many Q heads share each KV head
        int headSize = config.headSize();
        int contextLength = config.contextLength();

        if (DEBUG && layer == 0) {
            System.err.printf("[GQA] Config: nHeads=%d, nKVHeads=%d, kvMul=%d, headSize=%d\n",
                            nHeads, nKVHeads, kvMul, headSize);
        }

        // Compute Q (full dimension)
        weights.wq[layer].matmul(state.xb2, state.q, dim, dim);

        // Compute K and V (reduced dimension for GQA)
        // CRITICAL: K and V have different dimensions in GQA
        weights.wk[layer].matmul(state.xb2, state.k, kvDim, dim);
        weights.wv[layer].matmul(state.xb2, state.v, kvDim, dim);

        // Apply RoPE to Q and K
        applyRoPE(state.q, position, config.ropeTheta(), headSize, nHeads);
        applyRoPE(state.k, position, config.ropeTheta(), headSize, nKVHeads);

        // Cache K and V at current position
        // FIXED: Store in layout [position][kvHead][headSize] to match retrieval
        for (int kvHead = 0; kvHead < nKVHeads; kvHead++) {
            for (int i = 0; i < headSize; i++) {
                int srcIdx = kvHead * headSize + i;
                int dstIdx = position * nKVHeads * headSize + kvHead * headSize + i;
                state.keyCache[layer].setFloat(dstIdx, state.k.getFloat(srcIdx));
                state.valueCache[layer].setFloat(dstIdx, state.v.getFloat(srcIdx));
            }
        }

        if (DEBUG && layer == 0) {
            System.err.printf("[GQA] Cached K/V at position %d, layout: [pos=%d][kvHeads=%d][headSize=%d]\n",
                            position, position, nKVHeads, headSize);
        }

        // Multi-head attention with GQA
        // Each Q head processes independently, but KV heads are shared
        for (int h = 0; h < nHeads; h++) {
            int kvHead = h / kvMul; // Which KV head this Q head uses (0-3 for Granite 2B)

            if (DEBUG && layer == 0 && h < 4) {
                System.err.printf("[GQA] Head %d uses KV head %d (kvMul=%d)\n", h, kvHead, kvMul);
            }

            // Compute attention scores for this head
            for (int t = 0; t <= position; t++) {
                float score = 0.0f;

                // Dot product between Q[h] and K[kvHead] at position t
                // FIXED: Correct KV cache indexing for GQA layout
                for (int i = 0; i < headSize; i++) {
                    float qVal = state.q.getFloat(h * headSize + i);
                    // CRITICAL FIX: KV cache layout is [position][kvHead][headSize]
                    int kIdx = t * nKVHeads * headSize + kvHead * headSize + i;
                    float kVal = state.keyCache[layer].getFloat(kIdx);
                    score += qVal * kVal;
                }

                // Scale by sqrt(headSize)
                score /= (float) Math.sqrt(headSize);

                // Store attention score
                state.att.setFloat(h * contextLength + t, score);
            }

            // Apply causal mask (future positions get -inf)
            for (int t = position + 1; t < contextLength; t++) {
                state.att.setFloat(h * contextLength + t, Float.NEGATIVE_INFINITY);
            }

            // Softmax over valid positions
            softmax(state.att, h * contextLength, position + 1);

            // Weighted sum of values using shared KV heads
            for (int i = 0; i < headSize; i++) {
                float output = 0.0f;
                for (int t = 0; t <= position; t++) {
                    float weight = state.att.getFloat(h * contextLength + t);
                    // CRITICAL FIX: Same layout fix for value cache
                    int vIdx = t * nKVHeads * headSize + kvHead * headSize + i;
                    float vVal = state.valueCache[layer].getFloat(vIdx);
                    output += weight * vVal;
                }
                // FIXED: Output indexing for Q heads
                state.xb.setFloat(h * headSize + i, output);
            }
        }

        // Final output projection
        weights.wo[layer].matmul(state.xb, state.xb2, dim, dim);

        if (DEBUG && layer == 0) {
            System.err.printf("[GQA] Attention complete for layer %d\n", layer);
        }
    }

    /**
     * SwiGLU activation function used by Granite.
     * SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x)
     * Standard formulation used in Llama, Mistral, and other modern models.
     */
    private static void computeSwiGLU(State state, StandardWeights weights, int layer,
                                      GraniteConfiguration config) {
        int dim = config.dim();
        int hiddenDim = config.hiddenDim();

        // 1. Gate projection (w1)
        weights.w1[layer].matmul(state.xb2, state.hb, hiddenDim, dim);

        // 2. Up projection (w3)
        weights.w3[layer].matmul(state.xb2, state.hb2, hiddenDim, dim);

        // 3. Apply SiLU to gate projection
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        for (int i = 0; i < hiddenDim; i++) {
            float val = state.hb.getFloat(i);
            state.hb.setFloat(i, silu(val));
        }

        // 4. Element-wise multiplication: silu(gate) * up (standard SwiGLU formulation)
        for (int i = 0; i < hiddenDim; i++) {
            float siluGate = state.hb.getFloat(i);  // Already SiLU-activated
            float up = state.hb2.getFloat(i);       // Raw up projection
            state.hb.setFloat(i, siluGate * up);
        }

        // 5. Down projection (w2)
        weights.w2[layer].matmul(state.hb, state.xb2, dim, hiddenDim);

        if (DEBUG && layer == 0) {
            System.err.printf("[SwiGLU] FFN complete for layer %d\n", layer);
        }
    }

    /**
     * SiLU (Sigmoid Linear Unit) activation.
     * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     */
    private static float silu(float x) {
        // Numerical stability for large negative values
        if (x < -20.0f) {
            return 0.0f;
        }
        // Numerical stability for large positive values
        if (x > 20.0f) {
            return x;
        }
        return x / (1.0f + (float) Math.exp(-x));
    }

    /**
     * Apply Rotary Position Embeddings (RoPE).
     */
    private static void applyRoPE(FloatTensor tensor, int position, float theta,
                                 int headSize, int nHeads) {
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < headSize / 2; i++) {
                float freq = 1.0f / (float) Math.pow(theta, 2.0f * i / headSize);
                float angle = position * freq;
                float cos = (float) Math.cos(angle);
                float sin = (float) Math.sin(angle);

                int idx1 = h * headSize + i;
                int idx2 = h * headSize + i + headSize / 2;

                float v1 = tensor.getFloat(idx1);
                float v2 = tensor.getFloat(idx2);

                // Rotate the complex number
                tensor.setFloat(idx1, v1 * cos - v2 * sin);
                tensor.setFloat(idx2, v1 * sin + v2 * cos);
            }
        }
    }

    /**
     * Softmax activation over a range.
     */
    private static void softmax(FloatTensor x, int offset, int size) {
        // Find max for numerical stability
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            float val = x.getFloat(offset + i);
            if (val > max) {
                max = val;
            }
        }

        // Exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = x.getFloat(offset + i);
            float expVal = (float) Math.exp(val - max);
            x.setFloat(offset + i, expVal);
            sum += expVal;
        }

        // Normalize
        if (sum > 0) {
            for (int i = 0; i < size; i++) {
                x.setFloat(offset + i, x.getFloat(offset + i) / sum);
            }
        }
    }

    /**
     * Compute L2 norm for debugging.
     */
    private static float computeNorm(FloatTensor tensor, int size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = tensor.getFloat(i);
            sum += val * val;
        }
        return (float) Math.sqrt(sum / size);
    }

    /**
     * Validate tensor contains no NaN or Inf values.
     */
    private static void validateNoNaN(FloatTensor tensor, String name) {
        if (!VALIDATE_NAN) return;

        int size = Math.min(tensor.size(), 1000); // Check first 1000 elements
        for (int i = 0; i < size; i++) {
            float val = tensor.getFloat(i);
            if (Float.isNaN(val) || Float.isInfinite(val)) {
                throw new RuntimeException(
                    String.format("NaN/Inf detected in %s at index %d: value=%f", name, i, val));
            }
        }
    }
}