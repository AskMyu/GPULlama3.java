package org.beehive.gpullama3.model.moe;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Standard implementation of an Expert for MoE models.
 * 
 * Implements a feed-forward network with configurable activation function.
 * Supports lazy loading/unloading of weights for memory efficiency.
 */
public class StandardExpert implements Expert {
    
    private final int expertId;
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    
    // Expert weights (lazily loaded)
    private FloatTensor upWeights;
    private FloatTensor downWeights;
    private FloatTensor gateWeights; // For SwiGLU/GLU variants
    
    // TornadoVM arrays for GPU computation
    private FloatArray upWeightsArray;
    private FloatArray downWeightsArray;
    private FloatArray gateWeightsArray;
    
    // Intermediate buffers
    private FloatArray hiddenBuffer;
    private FloatArray gateBuffer;
    
    private boolean isLoaded = false;
    
    public StandardExpert(int expertId, int inputDim, int hiddenDim, int outputDim) {
        this.expertId = expertId;
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
    }
    
    @Override
    public void forward(FloatArray input, FloatArray output, String activationFunction) {
        if (!isLoaded) {
            throw new IllegalStateException("Expert " + expertId + " weights not loaded");
        }
        
        int inputSize = input.getSize() / inputDim;
        
        // Allocate intermediate buffers if needed
        ensureBuffersAllocated(inputSize);
        
        // Apply activation function variant
        switch (activationFunction.toLowerCase()) {
            case "swiglu" -> forwardSwiGLU(input, output, inputSize);
            case "gelu" -> forwardGELU(input, output, inputSize);
            case "relu" -> forwardReLU(input, output, inputSize);
            default -> throw new UnsupportedOperationException("Unsupported activation: " + activationFunction);
        }
    }
    
    /**
     * Forward pass with SwiGLU activation: gate(x) * swish(up(x))
     */
    private void forwardSwiGLU(FloatArray input, FloatArray output, int inputSize) {
        if (gateWeightsArray == null) {
            throw new IllegalStateException("SwiGLU requires gate weights");
        }
        
        // Compute up projection: up(x)
        matmul(input, upWeightsArray, hiddenBuffer, inputSize, inputDim, hiddenDim);
        
        // Compute gate projection: gate(x)
        matmul(input, gateWeightsArray, gateBuffer, inputSize, inputDim, hiddenDim);
        
        // Apply SwiGLU: gate(x) * swish(up(x))
        for (int i = 0; i < hiddenBuffer.getSize(); i++) {
            float upVal = hiddenBuffer.get(i);
            float gateVal = gateBuffer.get(i);

            // Swish/SiLU activation: x * sigmoid(x)
            float swishUp = upVal * (1.0f / (1.0f + (float) Math.exp(-upVal)));

            // Gate activation (typically no activation for gate)
            hiddenBuffer.set(i, gateVal * swishUp);
        }
        
        // Down projection: down(hidden)
        matmul(hiddenBuffer, downWeightsArray, output, inputSize, hiddenDim, outputDim);
    }
    
    /**
     * Forward pass with GELU activation
     */
    private void forwardGELU(FloatArray input, FloatArray output, int inputSize) {
        // Up projection
        matmul(input, upWeightsArray, hiddenBuffer, inputSize, inputDim, hiddenDim);
        
        // Apply GELU activation
        for (int i = 0; i < hiddenBuffer.getSize(); i++) {
            float x = hiddenBuffer.get(i);
            float gelu = 0.5f * x * (1.0f + (float) Math.tanh(Math.sqrt(2.0/Math.PI) * (x + 0.044715 * x * x * x)));
            hiddenBuffer.set(i, gelu);
        }
        
        // Down projection
        matmul(hiddenBuffer, downWeightsArray, output, inputSize, hiddenDim, outputDim);
    }
    
    /**
     * Forward pass with ReLU activation
     */
    private void forwardReLU(FloatArray input, FloatArray output, int inputSize) {
        // Up projection
        matmul(input, upWeightsArray, hiddenBuffer, inputSize, inputDim, hiddenDim);
        
        // Apply ReLU activation
        for (int i = 0; i < hiddenBuffer.getSize(); i++) {
            hiddenBuffer.set(i, Math.max(0.0f, hiddenBuffer.get(i)));
        }
        
        // Down projection
        matmul(hiddenBuffer, downWeightsArray, output, inputSize, hiddenDim, outputDim);
    }
    
    /**
     * Simple matrix multiplication implementation
     * TODO: Replace with optimized TornadoVM kernel
     */
    private void matmul(FloatArray a, FloatArray b, FloatArray result, int M, int K, int N) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a.get(m * K + k) * b.get(k * N + n);
                }
                result.set(m * N + n, sum);
            }
        }
    }
    
    private void ensureBuffersAllocated(int inputSize) {
        int hiddenSize = inputSize * hiddenDim;
        
        if (hiddenBuffer == null || hiddenBuffer.getSize() != hiddenSize) {
            hiddenBuffer = new FloatArray(hiddenSize);
        }
        
        if (gateWeightsArray != null && (gateBuffer == null || gateBuffer.getSize() != hiddenSize)) {
            gateBuffer = new FloatArray(hiddenSize);
        }
    }
    
    @Override
    public void loadWeights(FloatTensor upWeights, FloatTensor downWeights, FloatTensor gateWeights) {
        this.upWeights = upWeights;
        this.downWeights = downWeights;
        this.gateWeights = gateWeights;
        
        // Convert to TornadoVM arrays
        this.upWeightsArray = tensorToFloatArray(upWeights);
        this.downWeightsArray = tensorToFloatArray(downWeights);
        if (gateWeights != null) {
            this.gateWeightsArray = tensorToFloatArray(gateWeights);
        }
        
        this.isLoaded = true;
    }
    
    @Override
    public void unloadWeights() {
        this.upWeights = null;
        this.downWeights = null;
        this.gateWeights = null;
        this.upWeightsArray = null;
        this.downWeightsArray = null;
        this.gateWeightsArray = null;
        this.hiddenBuffer = null;
        this.gateBuffer = null;
        this.isLoaded = false;
    }
    
    private FloatArray tensorToFloatArray(FloatTensor tensor) {
        FloatArray array = new FloatArray(tensor.size());
        for (int i = 0; i < tensor.size(); i++) {
            array.set(i, tensor.getFloat(i));
        }
        return array;
    }
    
    @Override
    public int getExpertId() { return expertId; }
    
    @Override
    public int getInputDim() { return inputDim; }
    
    @Override
    public int getOutputDim() { return outputDim; }
    
    @Override
    public int getHiddenDim() { return hiddenDim; }
    
    @Override
    public boolean isLoaded() { return isLoaded; }
    
    @Override
    public long getMemoryUsage() {
        if (!isLoaded) return 0;
        
        long usage = 0;
        if (upWeightsArray != null) usage += upWeightsArray.getSize() * 4L; // 4 bytes per float
        if (downWeightsArray != null) usage += downWeightsArray.getSize() * 4L;
        if (gateWeightsArray != null) usage += gateWeightsArray.getSize() * 4L;
        if (hiddenBuffer != null) usage += hiddenBuffer.getSize() * 4L;
        if (gateBuffer != null) usage += gateBuffer.getSize() * 4L;
        
        return usage;
    }
}