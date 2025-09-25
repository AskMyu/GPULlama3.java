package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.inference.state.State;
import java.util.List;

/**
 * Interface for batch processing of prompt tokens.
 * Designed to solve expert routing context isolation in MoE models by processing
 * related tokens together to maintain contextual coherence.
 */
public interface BatchProcessor {

    /**
     * Process entire prompt in single batch operation.
     * This method should process all prompt tokens together to maintain context,
     * particularly important for MoE models where token-by-token processing
     * can fragment semantic relationships.
     *
     * @param state Model state to update with batch processing results
     * @param promptTokens All prompt tokens to process as a batch
     * @param startPosition Starting position in the sequence
     * @return Updated state after processing all prompt tokens
     */
    State processBatch(State state, List<Integer> promptTokens, int startPosition);

    /**
     * Check if this processor supports batch processing.
     *
     * @return true if batch processing is supported and functional
     */
    boolean supportsBatchProcessing();

    /**
     * Get the optimal batch size for this processor.
     *
     * @return Maximum number of tokens that can be efficiently processed in one batch
     */
    int getOptimalBatchSize();

    /**
     * Validate that the given tokens can be processed as a batch.
     *
     * @param tokens The tokens to validate
     * @return true if the tokens can be batch processed
     */
    default boolean validateBatch(List<Integer> tokens) {
        return tokens != null &&
               !tokens.isEmpty() &&
               tokens.size() <= getOptimalBatchSize() &&
               supportsBatchProcessing();
    }
}