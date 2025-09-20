package org.beehive.gpullama3.inference.weights.olmoe;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.LinkedHashMap;

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

    // SELECTIVE EXPERT LOADING: Source tensors for on-demand loading
    private final FloatTensor[] sourceExpertGateWeights;   // Original tensors from GGUF
    private final FloatTensor[] sourceExpertDownWeights;   // Original tensors from GGUF
    private final FloatTensor[] sourceExpertUpWeights;     // Original tensors from GGUF

    // EXPERT CACHE: LRU cache for frequently used experts
    private final Map<String, float[]> expertGateCache;    // "layer_expert" -> weights
    private final Map<String, float[]> expertDownCache;    // "layer_expert" -> weights
    private final Map<String, float[]> expertUpCache;      // "layer_expert" -> weights
    private final int maxCachedExperts = 32;               // Cache size limit

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
            FloatArray[] expertUpWeights,
            FloatTensor[] sourceExpertGateWeights,  // NEW: Source tensors for selective loading
            FloatTensor[] sourceExpertDownWeights,  // NEW: Source tensors for selective loading
            FloatTensor[] sourceExpertUpWeights) {  // NEW: Source tensors for selective loading

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

        // Store source tensors for selective expert loading
        this.sourceExpertGateWeights = sourceExpertGateWeights;
        this.sourceExpertDownWeights = sourceExpertDownWeights;
        this.sourceExpertUpWeights = sourceExpertUpWeights;

        // Initialize LRU caches for expert weights
        this.expertGateCache = new LinkedHashMap<String, float[]>(maxCachedExperts + 1, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                return size() > maxCachedExperts;
            }
        };
        this.expertDownCache = new LinkedHashMap<String, float[]>(maxCachedExperts + 1, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                return size() > maxCachedExperts;
            }
        };
        this.expertUpCache = new LinkedHashMap<String, float[]>(maxCachedExperts + 1, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                return size() > maxCachedExperts;
            }
        };

        System.err.printf("[SELECTIVE-EXPERT-LOADING] Initialized with cache size: %d experts per type\n", maxCachedExperts);
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

    /**
     * EXPERT WEIGHT ACCESS METHODS
     * These methods provide the interface expected by InferenceCore.forwardJavaOlmoe()
     * They bridge TornadoVM FloatArray to float[] array format used in CPU inference
     */

    public float[] getExpertGateWeights(int layer, int expertId, int numExperts, int hiddenDim, int dim) {
        // TEST: Swap dimensions - maybe weights stored as [dim, hiddenDim] not [hiddenDim, dim]
        return loadExpertWeights("gate", layer, expertId, dim, hiddenDim, expertGateCache,
                                sourceExpertGateWeights, expertGateWeights);
    }

    public float[] getExpertDownWeights(int layer, int expertId, int numExperts, int hiddenDim, int dim) {
        // TEST: Swap dimensions - maybe weights stored as [hiddenDim, dim] not [dim, hiddenDim]
        return loadExpertWeights("down", layer, expertId, hiddenDim, dim, expertDownCache,
                                sourceExpertDownWeights, expertDownWeights);
    }

    public float[] getExpertUpWeights(int layer, int expertId, int numExperts, int hiddenDim, int dim) {
        // TEST: Swap dimensions - maybe weights stored as [dim, hiddenDim] not [hiddenDim, dim]
        return loadExpertWeights("up", layer, expertId, dim, hiddenDim, expertUpCache,
                                sourceExpertUpWeights, expertUpWeights);
    }

    /**
     * SELECTIVE EXPERT LOADING CORE METHOD
     * Loads expert weights on-demand with intelligent caching
     */
    private float[] loadExpertWeights(String weightType, int layer, int expertId,
                                    int rows, int cols, Map<String, float[]> cache,
                                    FloatTensor[] sourceTensors, FloatArray[] fallbackArrays) {
        String cacheKey = layer + "_" + expertId;

        // Check cache first
        synchronized (cache) {
            float[] cached = cache.get(cacheKey);
            if (cached != null) {
                System.err.printf("[EXPERT-CACHE-HIT] %s layer=%d expert=%d (cache size: %d)\n",
                                weightType, layer, expertId, cache.size());
                return cached;
            }
        }

        // Cache miss - load from source tensor
        System.err.printf("[EXPERT-CACHE-MISS] Loading %s layer=%d expert=%d on-demand\n",
                        weightType, layer, expertId);

        float[] expertWeights = loadExpertFromSource(sourceTensors, fallbackArrays, layer, expertId, rows, cols);

        // Cache the loaded weights
        synchronized (cache) {
            cache.put(cacheKey, expertWeights);
            System.err.printf("[EXPERT-CACHED] %s layer=%d expert=%d (cache size: %d/%d)\n",
                            weightType, layer, expertId, cache.size(), maxCachedExperts);
        }

        return expertWeights;
    }

    /**
     * Load expert weights from source tensor or fallback to FloatArray
     */
    private float[] loadExpertFromSource(FloatTensor[] sourceTensors, FloatArray[] fallbackArrays,
                                       int layer, int expertId, int rows, int cols) {
        int expertSize = rows * cols;
        float[] expertWeights = new float[expertSize];

        // DEBUG: Always log what sources are available
        System.err.printf("[EXPERT-DEBUG] Loading layer=%d expert=%d, sourceTensors=%s, fallbackArrays=%s\n",
                         layer, expertId,
                         (sourceTensors != null && sourceTensors[layer] != null) ? "available" : "null",
                         (fallbackArrays != null && fallbackArrays[layer] != null) ? "available" : "null");

        // Try loading from source tensor first (better performance)
        if (sourceTensors != null && sourceTensors[layer] != null) {
            FloatTensor sourceTensor = sourceTensors[layer];

            // CRITICAL DEBUG: Log actual tensor dimensions to understand storage format
            int totalSize = (int)sourceTensor.size();
            System.err.printf("[TENSOR-DEBUG] Layer %d tensor total size: %d elements\n", layer, totalSize);
            System.err.printf("[TENSOR-DEBUG] Expected per expert: rows=%d, cols=%d, total=%d\n", rows, cols, expertSize);
            System.err.printf("[TENSOR-DEBUG] Expected 64 experts total: %d elements\n", 64 * expertSize);
            System.err.printf("[TENSOR-DEBUG] Actual vs Expected ratio: %.2f\n", (double)totalSize / (64 * expertSize));

            // CRITICAL TEST: Try simple contiguous layout (original approach)
            // The interleaved approach didn't work, so reverting to test basic extraction
            System.err.printf("[EXPERT-LAYOUT-TEST] Extracting expert %d with SIMPLE layout (rows=%d, cols=%d)\n",
                             expertId, rows, cols);

            int expertOffset = expertId * expertSize;

            // DEBUG: Sample first few weights to check values
            if (layer == 0 && expertId < 2) {
                System.err.printf("[WEIGHT-SAMPLE] Layer %d Expert %d first 5 weights:\n", layer, expertId);
                for (int i = 0; i < Math.min(5, expertSize); i++) {
                    float weight = sourceTensor.getFloat(expertOffset + i);
                    System.err.printf("  [%d] = %.6f\n", i, weight);
                }
            }

            for (int i = 0; i < expertSize; i++) {
                expertWeights[i] = sourceTensor.getFloat(expertOffset + i);
            }

            System.err.printf("[EXPERT-SOURCE-LOAD] Loaded from source tensor: %d weights (interleaved layout)\n", expertSize);
        }
        // Fallback to FloatArray if available
        else if (fallbackArrays != null && fallbackArrays[layer] != null) {
            FloatArray fallbackArray = fallbackArrays[layer];
            int expertOffset = expertId * expertSize;

            for (int i = 0; i < expertSize; i++) {
                expertWeights[i] = fallbackArray.get(expertOffset + i);
            }

            System.err.printf("[EXPERT-FALLBACK-LOAD] Loaded from FloatArray: %d weights\n", expertSize);
        }
        // Last resort: return zeros (will cause poor inference but won't crash)
        else {
            System.err.printf("[EXPERT-ZERO-FALLBACK] No source available - returning zeros for layer=%d expert=%d\n",
                            layer, expertId);
            // expertWeights is already initialized to zeros
        }

        return expertWeights;
    }
}