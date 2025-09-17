package org.beehive.gpullama3.model.moe;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Represents a single expert in a MoE (Mixture-of-Experts) layer.
 * 
 * Each expert is typically a feed-forward network that processes
 * a subset of the input tokens based on routing decisions.
 */
public interface Expert {
    
    /**
     * Processes input through this expert's computation.
     * 
     * @param input Input tensor for tokens assigned to this expert
     * @param output Output tensor after expert processing
     * @param activationFunction The activation function to use (e.g., "gelu", "swiglu", "relu")
     */
    void forward(FloatArray input, FloatArray output, String activationFunction);
    
    /**
     * Gets the expert's unique identifier.
     */
    int getExpertId();
    
    /**
     * Gets the input dimension for this expert.
     */
    int getInputDim();
    
    /**
     * Gets the output dimension for this expert.
     */
    int getOutputDim();
    
    /**
     * Gets the hidden dimension of this expert's FFN.
     */
    int getHiddenDim();
    
    /**
     * Loads the expert's weights from tensor entries.
     * This supports lazy loading of expert weights to save memory.
     * 
     * @param upWeights The "up" projection weights
     * @param downWeights The "down" projection weights  
     * @param gateWeights The "gate" projection weights (for SwiGLU/GLU variants)
     */
    void loadWeights(FloatTensor upWeights, FloatTensor downWeights, FloatTensor gateWeights);
    
    /**
     * Unloads expert weights from memory to free up space.
     * Used for dynamic expert loading/unloading in memory-constrained environments.
     */
    void unloadWeights();
    
    /**
     * Checks if this expert's weights are currently loaded in memory.
     */
    boolean isLoaded();
    
    /**
     * Gets the memory usage of this expert in bytes.
     */
    long getMemoryUsage();
}