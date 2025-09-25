package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.olmoe.OlmoeConfiguration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * State for OLMoE model with MoE-specific tracking.
 * 
 * Extends base State with additional tracking for expert routing,
 * load balancing, and auxiliary losses.
 */
public class OlmoeState extends State {
    
    // MoE-specific state
    public FloatArray[] expertRouterLogits;  // Router logits for each layer [layers][seq_len, num_experts]
    public IntArray[] selectedExperts;       // Selected experts per token per layer [layers][seq_len, top_k]
    public FloatArray[] expertWeights;       // Weights for selected experts [layers][seq_len, top_k]
    public FloatArray[] expertOutputs;       // Cached expert outputs for aggregation
    
    // Load balancing tracking
    public IntArray[] expertLoadCounts;      // Number of tokens routed to each expert
    public FloatArray routerAuxLoss;         // Auxiliary loss for load balancing

    // EXPERT CONSISTENCY: Batch processing support for solving context isolation
    public boolean batchExpertConsistencyMode = false;    // Enable shared expert routing
    public boolean[] sharedExpertsEstablished;            // Flag per layer if shared experts are set
    public int[][] sharedExpertsPerLayer;                 // Shared experts per layer [layers][top_k]
    public float[][] sharedExpertWeightsPerLayer;         // Shared expert weights per layer [layers][top_k]
    
    // Configuration
    private final OlmoeConfiguration config;
    
    public OlmoeState(Configuration configuration, int batchsize) {
        super(configuration, batchsize);
        
        if (!(configuration instanceof OlmoeConfiguration)) {
            throw new IllegalArgumentException("OlmoeState requires OlmoeConfiguration");
        }
        
        this.config = (OlmoeConfiguration) configuration;
        
        // Initialize MoE-specific arrays
        initializeMoEState();
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        OlmoeConfiguration olmoeConfig = (OlmoeConfiguration) config;
        StateFields fields = new StateFields();

        // Allocation with OLMoE dimensions
        fields.x = ArrayFloatTensor.allocate(olmoeConfig.dim());
        fields.xb = ArrayFloatTensor.allocate(olmoeConfig.dim());
        fields.xb2 = ArrayFloatTensor.allocate(olmoeConfig.dim());
        fields.hb = ArrayFloatTensor.allocate(olmoeConfig.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(olmoeConfig.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(olmoeConfig.dim());
        fields.k = ArrayFloatTensor.allocate(olmoeConfig.dim());
        fields.v = ArrayFloatTensor.allocate(olmoeConfig.dim());
        fields.att = ArrayFloatTensor.allocate(olmoeConfig.numberOfHeads(), olmoeConfig.contextLength());
        fields.logits = ArrayFloatTensor.allocate(olmoeConfig.vocabularySize());

        // Key-value cache with OLMoE dimensions
        int kvDim = (olmoeConfig.dim() * olmoeConfig.numberOfKeyValueHeads()) / olmoeConfig.numberOfHeads();
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(olmoeConfig.contextLength(), kvDim)).limit(olmoeConfig.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(olmoeConfig.contextLength(), kvDim)).limit(olmoeConfig.numberOfLayers()).toArray(FloatTensor[]::new);

        // TornadoVM wrappers with OLMoE dimensions
        fields.wrapX = new FloatArray(olmoeConfig.dim());
        fields.wrapXb = new FloatArray(olmoeConfig.dim());
        fields.wrapXb2 = new FloatArray(olmoeConfig.dim());
        fields.wrapHb = new FloatArray(olmoeConfig.hiddenDim());
        fields.wrapHb2 = new FloatArray(olmoeConfig.hiddenDim());
        fields.wrapLogits = new FloatArray(olmoeConfig.vocabularySize());
        fields.wrapQ = new FloatArray(olmoeConfig.dim());
        fields.wrapK = new FloatArray(olmoeConfig.dim());
        fields.wrapV = new FloatArray(olmoeConfig.dim());
        fields.wrapAtt = new FloatArray(olmoeConfig.numberOfHeads() * olmoeConfig.contextLength());

        // KV cache wrappers
        fields.wrapKeyCache = new FloatArray(olmoeConfig.numberOfLayers() * olmoeConfig.contextLength() * kvDim);
        fields.wrapValueCache = new FloatArray(olmoeConfig.numberOfLayers() * olmoeConfig.contextLength() * kvDim);

        // Position holder and temporary arrays
        fields.positionHolder = new IntArray(1);
        fields.temp = new FloatArray(Math.max(9, olmoeConfig.numberOfHeads()));
        fields.tempFFN = new FloatArray(olmoeConfig.hiddenDim());
        fields.tempLogits = new FloatArray(olmoeConfig.vocabularySize());

        return fields;
    }
    
    private void initializeMoEState() {
        int numLayers = config.numberOfLayers();
        int numExperts = config.getNumberOfExperts();
        int topK = config.getNumberOfActiveExperts();
        int maxSeqLen = config.contextLength();
        
        // Initialize router state arrays
        expertRouterLogits = new FloatArray[numLayers];
        selectedExperts = new IntArray[numLayers];
        expertWeights = new FloatArray[numLayers];
        expertOutputs = new FloatArray[numLayers];
        expertLoadCounts = new IntArray[numLayers];
        
        // Initialize expert consistency arrays
        sharedExpertsEstablished = new boolean[numLayers];
        sharedExpertsPerLayer = new int[numLayers][topK];
        sharedExpertWeightsPerLayer = new float[numLayers][topK];

        for (int i = 0; i < numLayers; i++) {
            // Router logits for all experts
            expertRouterLogits[i] = new FloatArray(maxSeqLen * numExperts);
            
            // Selected top-K experts per token
            selectedExperts[i] = new IntArray(maxSeqLen * topK);
            
            // Weights for selected experts
            expertWeights[i] = new FloatArray(maxSeqLen * topK);
            
            // Expert outputs cache
            expertOutputs[i] = new FloatArray(maxSeqLen * config.dim());
            
            // Load balancing counters
            expertLoadCounts[i] = new IntArray(numExperts);
        }
        
        // Auxiliary loss tracking
        routerAuxLoss = new FloatArray(1);
    }
    
    /**
     * Resets MoE state for a new sequence
     */
    public void resetMoEState() {
        if (expertRouterLogits == null) {
            return;
        }

        // Clear expert selection and weights
        for (int i = 0; i < config.numberOfLayers(); i++) {
            expertRouterLogits[i].init(0.0f);
            selectedExperts[i].init(0);
            expertWeights[i].init(0.0f);
            expertOutputs[i].init(0.0f);
            expertLoadCounts[i].init(0);
        }

        routerAuxLoss.init(0.0f);
    }

    /**
     * CRITICAL FIX: Clears KV cache for new sequence to prevent context contamination
     * This resolves the issue where model generates identical output regardless of input prompt
     */
    public void clearKVCache() {
        System.out.println("[OLMOE-CACHE-CLEAR] 🧹 Clearing KV cache to prevent context contamination");

        // Clear KV cache arrays to prevent old context from contaminating new generation
        // This was the root cause of identical outputs regardless of input prompt
        if (wrapKeyCache instanceof FloatArray keyCache) {
            keyCache.init(0.0f);
            System.out.println("[OLMOE-CACHE-CLEAR] ✅ Key cache cleared (" + keyCache.getSize() + " elements)");
        } else if (wrapKeyCache != null) {
            System.out.println("[OLMOE-CACHE-CLEAR] ⚠️ Key cache is SmartCacheArray - using reflection to clear");
            try {
                // Use reflection for SmartCacheArray
                wrapKeyCache.getClass().getMethod("init", float.class).invoke(wrapKeyCache, 0.0f);
                System.out.println("[OLMOE-CACHE-CLEAR] ✅ Key cache cleared via reflection");
            } catch (Exception e) {
                System.out.println("[OLMOE-CACHE-CLEAR] ❌ Failed to clear key cache: " + e.getMessage());
            }
        }

        if (wrapValueCache instanceof FloatArray valueCache) {
            valueCache.init(0.0f);
            System.out.println("[OLMOE-CACHE-CLEAR] ✅ Value cache cleared (" + valueCache.getSize() + " elements)");
        } else if (wrapValueCache != null) {
            System.out.println("[OLMOE-CACHE-CLEAR] ⚠️ Value cache is SmartCacheArray - using reflection to clear");
            try {
                // Use reflection for SmartCacheArray
                wrapValueCache.getClass().getMethod("init", float.class).invoke(wrapValueCache, 0.0f);
                System.out.println("[OLMOE-CACHE-CLEAR] ✅ Value cache cleared via reflection");
            } catch (Exception e) {
                System.out.println("[OLMOE-CACHE-CLEAR] ❌ Failed to clear value cache: " + e.getMessage());
            }
        }

        // Also clear MoE state for completeness
        resetMoEState();
        System.out.println("[OLMOE-CACHE-CLEAR] ✅ MoE state cleared");

        System.out.println("[OLMOE-CACHE-CLEAR] 🎯 Complete cache reset - ready for new context");
    }
    
    /**
     * Updates load balancing statistics after routing
     */
    public void updateLoadBalancing(int layer, IntArray selectedExpertsForLayer) {
        if (expertLoadCounts == null) {
            return;
        }
        
        // Count how many tokens went to each expert
        for (int i = 0; i < selectedExpertsForLayer.getSize(); i++) {
            int expertId = selectedExpertsForLayer.get(i);
            if (expertId >= 0 && expertId < config.getNumberOfExperts()) {
                int currentCount = expertLoadCounts[layer].get(expertId);
                expertLoadCounts[layer].set(expertId, currentCount + 1);
            }
        }
    }
    
    /**
     * Calculates auxiliary loss for load balancing
     */
    public float calculateAuxiliaryLoss() {
        if (expertLoadCounts == null) {
            return 0.0f;
        }
        
        float totalLoss = 0.0f;
        int numExperts = config.getNumberOfExperts();
        
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            // Calculate load imbalance for this layer
            int totalTokens = 0;
            for (int e = 0; e < numExperts; e++) {
                totalTokens += expertLoadCounts[layer].get(e);
            }
            
            if (totalTokens > 0) {
                float idealLoad = (float) totalTokens / numExperts;
                float imbalance = 0.0f;
                
                for (int e = 0; e < numExperts; e++) {
                    float load = expertLoadCounts[layer].get(e);
                    imbalance += Math.abs(load - idealLoad);
                }
                
                totalLoss += imbalance / totalTokens;
            }
        }
        
        return totalLoss * config.getRouterAuxLossCoef();
    }

    /**
     * Adds auxiliary losses for a specific layer during MoE routing.
     *
     * @param layer Layer index
     * @param loadBalancingLoss Load balancing auxiliary loss value
     * @param routerZLoss Router z-loss value
     */
    public void addAuxiliaryLoss(int layer, float loadBalancingLoss, float routerZLoss) {
        // During inference, we compute but don't use auxiliary losses
        // They would be used during training to prevent router collapse
        float combinedLoss = loadBalancingLoss * 0.01f + routerZLoss * 0.001f; // OLMoE coefficients

        // Store auxiliary loss for monitoring (during inference, not used for training)
        // During training, these losses would be added to the main loss function

        // Log auxiliary loss values for monitoring router health
        if (layer == 0) { // Log only for first layer to avoid spam
            System.err.printf("[MOE-AUX-LOSS] Layer %d: LoadBalance=%.6f, RouterZ=%.6f, Combined=%.6f%n",
                             layer, loadBalancingLoss, routerZLoss, combinedLoss);
        }
    }

}