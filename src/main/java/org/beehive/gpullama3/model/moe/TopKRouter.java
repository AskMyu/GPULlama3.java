package org.beehive.gpullama3.model.moe;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Top-K router implementation for MoE models.
 * 
 * Routes each token to the top-K experts based on routing scores.
 * This is the standard routing strategy used in most MoE models including GPT-OSS.
 */
public class TopKRouter implements ExpertRouter {
    
    private final RoutingConfig config;
    
    public TopKRouter(RoutingConfig config) {
        this.config = config;
    }
    
    @Override
    public void route(FloatArray input, FloatArray routingWeights,
                      IntArray selectedExperts, FloatArray expertWeights,
                      RoutingConfig config) {
        
        int batchSize = 1; // Assuming batch size of 1 for now
        int seqLength = input.getSize() / getHiddenSize(input, routingWeights);
        int hiddenSize = getHiddenSize(input, routingWeights);
        int numExperts = config.numExperts();
        int topK = config.activeExperts();
        
        // Process each token in the sequence
        for (int seq = 0; seq < seqLength; seq++) {
            int inputOffset = seq * hiddenSize;
            
            // Compute routing scores: input * routing_weights
            float[] scores = new float[numExperts];
            for (int expert = 0; expert < numExperts; expert++) {
                float score = 0.0f;
                for (int h = 0; h < hiddenSize; h++) {
                    score += input.get(inputOffset + h) * routingWeights.get(h * numExperts + expert);
                }
                scores[expert] = score;
            }
            
            // Apply softmax to get probabilities
            applySoftmax(scores);
            
            // Select top-K experts
            int[] topKIndices = selectTopK(scores, topK);
            float[] topKWeights = new float[topK];
            
            // Extract weights for selected experts and renormalize
            float weightSum = 0.0f;
            for (int k = 0; k < topK; k++) {
                topKWeights[k] = scores[topKIndices[k]];
                weightSum += topKWeights[k];
            }
            
            // Normalize weights so they sum to 1
            if (weightSum > 0.0f) {
                for (int k = 0; k < topK; k++) {
                    topKWeights[k] /= weightSum;
                }
            }
            
            // Store results
            int outputOffset = seq * topK;
            for (int k = 0; k < topK; k++) {
                selectedExperts.set(outputOffset + k, topKIndices[k]);
                expertWeights.set(outputOffset + k, topKWeights[k]);
            }
        }
    }
    
    @Override
    public RoutingConfig getConfig() {
        return config;
    }
    
    /**
     * Applies softmax to convert scores to probabilities.
     */
    private void applySoftmax(float[] scores) {
        // Find max for numerical stability
        float maxScore = Float.NEGATIVE_INFINITY;
        for (float score : scores) {
            maxScore = Math.max(maxScore, score);
        }
        
        // Compute exp(score - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(scores[i] - maxScore);
            sum += scores[i];
        }
        
        // Normalize
        for (int i = 0; i < scores.length; i++) {
            scores[i] /= sum;
        }
    }
    
    /**
     * Selects the top-K experts based on their scores.
     */
    private int[] selectTopK(float[] scores, int k) {
        int[] indices = new int[k];
        boolean[] used = new boolean[scores.length];
        
        for (int selected = 0; selected < k; selected++) {
            int bestIdx = -1;
            float bestScore = Float.NEGATIVE_INFINITY;
            
            for (int i = 0; i < scores.length; i++) {
                if (!used[i] && scores[i] > bestScore) {
                    bestScore = scores[i];
                    bestIdx = i;
                }
            }
            
            if (bestIdx >= 0) {
                indices[selected] = bestIdx;
                used[bestIdx] = true;
            }
        }
        
        return indices;
    }
    
    /**
     * Infers hidden size from input and routing weights dimensions.
     */
    private int getHiddenSize(FloatArray input, FloatArray routingWeights) {
        // routing_weights shape: [hidden_size, num_experts]
        return routingWeights.getSize() / config.numExperts();
    }
}