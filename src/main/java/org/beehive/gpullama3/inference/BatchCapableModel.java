package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.inference.state.State;
import java.util.List;

/**
 * Interface for models that support native batch processing capabilities.
 * Models implementing this interface can process multiple tokens simultaneously
 * to maintain better context coherence, particularly important for MoE models
 * where token-by-token processing can fragment semantic relationships.
 */
public interface BatchCapableModel {

    /**
     * Process multiple tokens in a single forward pass.
     * This method should handle all aspects of batch processing including:
     * - Batch embedding lookup
     * - Batch attention computation
     * - Batch expert routing (for MoE models)
     * - KV cache updates for all positions
     *
     * @param state Current model state to be updated
     * @param tokens Array of tokens to process together
     * @param startPosition Starting sequence position for the batch
     * @return Updated state with batch processing results
     */
    State forwardBatch(State state, List<Integer> tokens, int startPosition);

    /**
     * Get the optimal batch size for this model.
     * This should return the maximum number of tokens that can be efficiently
     * processed together considering memory constraints and model architecture.
     *
     * @return Optimal batch size for this model
     */
    int getOptimalBatchSize();

    /**
     * Check if batch processing is currently available and functional.
     * This can return false if there are temporary issues like GPU memory
     * constraints or if required components are not initialized.
     *
     * @return true if batch processing is currently functional
     */
    boolean isBatchProcessingAvailable();

    /**
     * Get the memory requirements for processing a batch of given size.
     * This helps the inference engine decide whether to use batch processing
     * or fall back to serial processing based on available memory.
     *
     * @param batchSize The size of the batch to estimate memory for
     * @return Estimated memory requirement in bytes
     */
    default long estimateBatchMemoryRequirement(int batchSize) {
        // Default conservative estimate: assume each token needs 4KB of working memory
        return batchSize * 4096L;
    }

    /**
     * Prepare the model for batch processing with the given batch size.
     * This method can be used to pre-allocate memory or initialize
     * batch-specific data structures.
     *
     * @param maxBatchSize Maximum batch size that will be processed
     * @return true if preparation was successful
     */
    default boolean prepareBatchProcessing(int maxBatchSize) {
        // Default implementation - no special preparation needed
        return true;
    }

    /**
     * Clean up any batch-specific resources.
     * This method should be called when batch processing is no longer needed
     * to free up memory and other resources.
     */
    default void cleanupBatchProcessing() {
        // Default implementation - no cleanup needed
    }
}