package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.CacheAllocationStrategy;
import org.beehive.gpullama3.tornadovm.SmartCacheArray;
import org.beehive.gpullama3.tornadovm.CacheTopology;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * Represents the state of the Llama model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Llama model.
 *
 * <p><b>Note 1:</b> LlamaState contains additional fields for TornadoVM wrappers
 * to enable GPU-accelerated processing of the model.</p>
 *
 * <p><b>Note 2:</b> This state implementation is also used for the Mistral model.</p>
 */
public class LlamaState extends State {

    public LlamaState(Configuration config, int batchsize) {
        super(config, batchsize);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();

        // Allocation with Llama/Mistral dimensions
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.dim());
        fields.v = ArrayFloatTensor.allocate(config.dim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Llama/Mistral dimensions
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        
        System.out.println("LlamaState: Initializing cache for " + config.numberOfLayers() + " layers");
        System.out.println("  Model dimensions: dim=" + config.dim() + ", contextLength=" + config.contextLength() + 
                          ", kvDim=" + kvDim);
        
        long keyCacheSize = (long) config.contextLength() * kvDim * config.numberOfLayers();
        long valueCacheSize = keyCacheSize; // Same size for key and value caches
        
        // Create layer-wise cache arrays (ArrayFloatTensor) for regular tensor operations
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                                .limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                                 .limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        
        // Phase 2: Use SmartCacheArray for unlimited cache size support
        // Determine if this is a VLM model (check for vision-related configuration)
        boolean isVLM = config.toString().toLowerCase().contains("llava") || 
                       config.toString().toLowerCase().contains("vlm") ||
                       System.getProperty("llava.token.reduction.enable", "false").equals("true");
        
        // Create cache topology for optimization
        CacheTopology topology = isVLM ? 
            new CacheTopology(config.numberOfLayers(), config.contextLength(), kvDim, true, 576) :
            new CacheTopology(config.numberOfLayers(), config.contextLength(), kvDim);
        
        // Use SmartCacheArray for automatic >2GB handling
        fields.wrapKeyCache = new SmartCacheArray((int) keyCacheSize, topology);
        fields.wrapValueCache = new SmartCacheArray((int) valueCacheSize, topology);
        
        System.out.printf("[LLAMA-STATE] Cache allocation: %s model with %.1f MB per cache%n",
                        isVLM ? "VLM" : "LLM", (keyCacheSize * 4.0) / (1024 * 1024));
        
        // Keep Phase 4L fallback available as backup
        if (!((SmartCacheArray)fields.wrapKeyCache).isBatched() && 
            !((SmartCacheArray)fields.wrapValueCache).isBatched()) {
            System.out.println("✓ Using standard allocation (model within 2GB limit)");
        } else {
            System.out.println("🚀 Using Phase 2 batched allocation (unlimited model size support)");
            topology.printAnalysis();
        }

        // TornadoVM wrappers with Llama/Mistral dimensions - use standard FloatArray for smaller allocations
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(config.dim());
        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(config.dim());
        fields.wrapK = new FloatArray(config.dim());
        fields.wrapV = new FloatArray(config.dim());
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // Temporary arrays for intermediate calculations
        fields.temp = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));

        return fields;
    }
}
