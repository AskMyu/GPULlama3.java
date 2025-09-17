package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.gptoss.GptOssConfiguration;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

public class GptOssState extends State {
    
    // MoE-specific state
    private final FloatArray[] expertSelections;
    private final FloatArray[] expertWeights;
    private final FloatArray routingBuffer;
    
    public GptOssState(GptOssConfiguration config) {
        super(config, 1); // batchsize = 1
        
        int numMoeLayers = config.numberOfLayers(); // Assume all layers can be MoE for simplicity
        
        // Initialize MoE-specific state
        this.expertSelections = new FloatArray[numMoeLayers];
        this.expertWeights = new FloatArray[numMoeLayers];
        
        int maxSequenceLength = config.contextLength();
        for (int i = 0; i < numMoeLayers; i++) {
            this.expertSelections[i] = new FloatArray(maxSequenceLength * config.activeExperts());
            this.expertWeights[i] = new FloatArray(maxSequenceLength * config.activeExperts());
        }
        
        this.routingBuffer = new FloatArray(config.dim() * config.numExperts());
    }
    
    @Override
    protected StateFields createStateFields(Configuration config) {
        GptOssConfiguration gptOssConfig = (GptOssConfiguration) config;
        StateFields fields = new StateFields();
        
        int dim = gptOssConfig.dim();
        int hiddenDim = gptOssConfig.hiddenDim();
        int numberOfLayers = gptOssConfig.numberOfLayers();
        int numberOfHeads = gptOssConfig.numberOfHeads();
        int numberOfKeyValueHeads = gptOssConfig.numberOfKeyValueHeads();
        int contextLength = gptOssConfig.contextLength();
        int vocabSize = gptOssConfig.vocabularySize();
        int kvDim = (dim * numberOfKeyValueHeads) / numberOfHeads;
        
        // Allocate tensors using ArrayFloatTensor
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(dim);
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(hiddenDim);
        fields.hb2 = ArrayFloatTensor.allocate(hiddenDim);
        fields.q = ArrayFloatTensor.allocate(dim);
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(numberOfHeads, contextLength);
        fields.logits = ArrayFloatTensor.allocate(vocabSize);
        
        // KV cache
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(contextLength, kvDim))
                .limit(numberOfLayers)
                .toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(contextLength, kvDim))
                .limit(numberOfLayers)
                .toArray(FloatTensor[]::new);
        
        // TornadoVM wrappers
        fields.wrapX = new FloatArray(dim);
        fields.wrapXb = new FloatArray(dim);
        fields.wrapXb2 = new FloatArray(dim);
        fields.wrapHb = new FloatArray(hiddenDim);
        fields.wrapHb2 = new FloatArray(hiddenDim);
        fields.wrapQ = new FloatArray(dim);
        fields.wrapK = new FloatArray(kvDim);
        fields.wrapV = new FloatArray(kvDim);
        fields.wrapAtt = new FloatArray(numberOfHeads * contextLength);
        fields.wrapLogits = new FloatArray(vocabSize);
        
        // KV cache wrappers
        int totalKvCacheSize = numberOfLayers * contextLength * kvDim;
        fields.wrapKeyCache = new FloatArray(totalKvCacheSize);
        fields.wrapValueCache = new FloatArray(totalKvCacheSize);
        
        fields.positionHolder = new IntArray(1);
        
        // Temporary buffers
        fields.temp = new FloatArray(1 + ((dim + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((hiddenDim + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((vocabSize + localSize - 1) / localSize));
        
        return fields;
    }
    
    // MoE-specific getters
    public FloatArray getExpertSelections(int layerIndex) {
        return expertSelections[layerIndex];
    }
    
    public FloatArray getExpertWeights(int layerIndex) {
        return expertWeights[layerIndex];
    }
    
    public FloatArray getRoutingBuffer() {
        return routingBuffer;
    }
    
    // GPU state management for TornadoVM
    public void prefetchToGPU() {
        // Mark arrays for GPU transfer
        // TornadoVM will handle the actual transfer
    }
    
    public void syncFromGPU() {
        // Synchronize state back from GPU
        // TornadoVM will handle the actual sync
    }
}