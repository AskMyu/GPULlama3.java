package org.beehive.gpullama3.model.moe;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Abstract interface for expert routing in MoE (Mixture-of-Experts) models.
 * 
 * The router determines which experts should process each token based on
 * the input representation and routing weights.
 */
public interface ExpertRouter {
    
    /**
     * Configuration for expert routing behavior.
     */
    record RoutingConfig(
        int numExperts,          // Total number of experts available
        int activeExperts,       // Number of experts to activate per token
        float routingNoise,      // Noise factor for training stability (0.0 for inference)
        boolean useLoadBalancing // Whether to apply load balancing losses
    ) {}
    
    /**
     * Routes input tokens to appropriate experts.
     * 
     * @param input Input tensor [batch_size, seq_length, hidden_size]
     * @param routingWeights Router weights [hidden_size, num_experts]
     * @param selectedExperts Output: indices of selected experts [batch_size, seq_length, active_experts]
     * @param expertWeights Output: weights for selected experts [batch_size, seq_length, active_experts]
     * @param config Routing configuration
     */
    void route(FloatArray input, FloatArray routingWeights,
               IntArray selectedExperts, FloatArray expertWeights,
               RoutingConfig config);
    
    /**
     * Gets the routing configuration for this router.
     */
    RoutingConfig getConfig();
    
    /**
     * Applies load balancing to encourage uniform expert utilization.
     * This is typically used during training to prevent expert collapse.
     * 
     * @param expertCounts Number of tokens routed to each expert
     * @param totalTokens Total number of tokens processed
     * @return Load balancing loss value
     */
    default float computeLoadBalancingLoss(IntArray expertCounts, int totalTokens) {
        if (expertCounts.getSize() == 0 || totalTokens == 0) return 0.0f;
        
        float targetLoad = (float) totalTokens / expertCounts.getSize();
        float loss = 0.0f;
        
        for (int i = 0; i < expertCounts.getSize(); i++) {
            float diff = expertCounts.get(i) - targetLoad;
            loss += diff * diff;
        }
        
        return loss / expertCounts.getSize();
    }
}