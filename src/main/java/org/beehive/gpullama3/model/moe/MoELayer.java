package org.beehive.gpullama3.model.moe;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Mixture-of-Experts (MoE) layer implementation.
 * 
 * Coordinates expert routing and computation for efficient sparse processing.
 * This is the main component that integrates routing and expert execution.
 */
public class MoELayer {
    
    private final ExpertRouter router;
    private final Expert[] experts;
    private final MoEConfig config;
    
    // Routing buffers
    private final IntArray selectedExperts;
    private final FloatArray expertWeights;
    private final FloatArray routingWeights;
    
    // Expert computation buffers
    private final FloatArray[] expertOutputs;
    private final FloatArray combinedOutput;
    
    /**
     * Configuration for the MoE layer.
     */
    public record MoEConfig(
        int numExperts,           // Total number of experts
        int activeExperts,        // Number of experts activated per token
        int inputDim,             // Input dimension
        int hiddenDim,            // Expert hidden dimension
        int outputDim,            // Output dimension
        String activationFunction, // Expert activation function ("swiglu", "gelu", "relu")
        boolean useLoadBalancing, // Whether to apply load balancing
        float routingNoise        // Routing noise for training (0.0 for inference)
    ) {}
    
    public MoELayer(MoEConfig config) {
        this.config = config;
        
        // Create router
        ExpertRouter.RoutingConfig routingConfig = new ExpertRouter.RoutingConfig(
            config.numExperts, config.activeExperts, 
            config.routingNoise, config.useLoadBalancing
        );
        this.router = new TopKRouter(routingConfig);
        
        // Create experts
        this.experts = new Expert[config.numExperts];
        for (int i = 0; i < config.numExperts; i++) {
            this.experts[i] = new StandardExpert(i, config.inputDim, config.hiddenDim, config.outputDim);
        }
        
        // Allocate routing buffers (sized for maximum possible usage)
        int maxTokens = 1024; // Reasonable default, can be adjusted
        this.selectedExperts = new IntArray(maxTokens * config.activeExperts);
        this.expertWeights = new FloatArray(maxTokens * config.activeExperts);
        this.routingWeights = new FloatArray(config.inputDim * config.numExperts);
        
        // Allocate expert output buffers
        this.expertOutputs = new FloatArray[config.numExperts];
        for (int i = 0; i < config.numExperts; i++) {
            this.expertOutputs[i] = new FloatArray(maxTokens * config.outputDim);
        }
        
        this.combinedOutput = new FloatArray(maxTokens * config.outputDim);
    }
    
    /**
     * Forward pass through the MoE layer.
     * 
     * @param input Input tensor [seq_length, input_dim]
     * @param output Output tensor [seq_length, output_dim]
     * @param sequenceLength Number of tokens in the sequence
     */
    public void forward(FloatArray input, FloatArray output, int sequenceLength) {
        // 1. Route tokens to experts
        router.route(input, routingWeights, selectedExperts, expertWeights, router.getConfig());
        
        // 2. Collect tokens for each expert
        groupTokensByExpert(input, sequenceLength);
        
        // 3. Process each expert's assigned tokens
        processExperts(sequenceLength);
        
        // 4. Combine expert outputs with routing weights
        combineExpertOutputs(output, sequenceLength);
    }
    
    /**
     * Groups input tokens by the experts they should be processed by.
     * This optimizes computation by batching tokens for each expert.
     */
    private void groupTokensByExpert(FloatArray input, int sequenceLength) {
        // For simplicity, we'll process tokens individually
        // A more optimized implementation would batch tokens by expert
        
        // Clear expert output buffers
        for (FloatArray expertOutput : expertOutputs) {
            for (int i = 0; i < expertOutput.getSize(); i++) {
                expertOutput.set(i, 0.0f);
            }
        }
    }
    
    /**
     * Processes tokens through their assigned experts.
     */
    private void processExperts(int sequenceLength) {
        int activeExperts = config.activeExperts;
        
        for (int seq = 0; seq < sequenceLength; seq++) {
            // Get input token
            FloatArray tokenInput = extractToken(seq);
            
            // Process through each active expert for this token
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selectedExperts.get(seq * activeExperts + k);
                float weight = expertWeights.get(seq * activeExperts + k);
                
                if (weight > 0.0f && experts[expertIdx].isLoaded()) {
                    // Process token through this expert
                    FloatArray expertOutput = new FloatArray(config.outputDim);
                    experts[expertIdx].forward(tokenInput, expertOutput, config.activationFunction);
                    
                    // Store weighted output
                    storeWeightedExpertOutput(seq, expertIdx, expertOutput, weight);
                }
            }
        }
    }
    
    /**
     * Combines expert outputs using routing weights to produce final output.
     */
    private void combineExpertOutputs(FloatArray output, int sequenceLength) {
        int activeExperts = config.activeExperts;
        int outputDim = config.outputDim;
        
        // Clear output
        for (int i = 0; i < output.getSize(); i++) {
            output.set(i, 0.0f);
        }
        
        // Combine outputs from all active experts
        for (int seq = 0; seq < sequenceLength; seq++) {
            for (int k = 0; k < activeExperts; k++) {
                int expertIdx = selectedExperts.get(seq * activeExperts + k);
                float weight = expertWeights.get(seq * activeExperts + k);
                
                if (weight > 0.0f) {
                    // Add this expert's weighted contribution
                    for (int dim = 0; dim < outputDim; dim++) {
                        int outputPos = seq * outputDim + dim;
                        int expertPos = seq * outputDim + dim;
                        
                        float expertValue = expertOutputs[expertIdx].get(expertPos);
                        output.set(outputPos, output.get(outputPos) + weight * expertValue);
                    }
                }
            }
        }
    }
    
    /**
     * Extracts a single token's input from the input tensor.
     */
    private FloatArray extractToken(int tokenIndex) {
        FloatArray token = new FloatArray(config.inputDim);
        // This is a simplified implementation - would need proper tensor extraction
        return token;
    }
    
    /**
     * Stores weighted expert output for later combination.
     */
    private void storeWeightedExpertOutput(int tokenIndex, int expertIdx, 
                                          FloatArray expertOutput, float weight) {
        for (int dim = 0; dim < config.outputDim; dim++) {
            int pos = tokenIndex * config.outputDim + dim;
            float weightedValue = expertOutput.get(dim) * weight;
            expertOutputs[expertIdx].set(pos, weightedValue);
        }
    }
    
    /**
     * Sets the routing weights for this layer.
     */
    public void setRoutingWeights(FloatArray weights) {
        if (weights.getSize() != routingWeights.getSize()) {
            throw new IllegalArgumentException("Routing weights size mismatch");
        }
        
        for (int i = 0; i < weights.getSize(); i++) {
            routingWeights.set(i, weights.get(i));
        }
    }
    
    /**
     * Gets the configuration for this MoE layer.
     */
    public MoEConfig getConfig() {
        return config;
    }
    
    /**
     * Gets all experts in this layer.
     */
    public Expert[] getExperts() {
        return experts;
    }
    
    /**
     * Gets the router used by this layer.
     */
    public ExpertRouter getRouter() {
        return router;
    }
    
    /**
     * Computes the total memory usage of this MoE layer.
     */
    public long getMemoryUsage() {
        long usage = 0;
        
        // Expert memory
        for (Expert expert : experts) {
            usage += expert.getMemoryUsage();
        }
        
        // Buffer memory
        usage += selectedExperts.getSize() * 4L;
        usage += expertWeights.getSize() * 4L;
        usage += routingWeights.getSize() * 4L;
        usage += combinedOutput.getSize() * 4L;
        
        for (FloatArray expertOutput : expertOutputs) {
            usage += expertOutput.getSize() * 4L;
        }
        
        return usage;
    }
}