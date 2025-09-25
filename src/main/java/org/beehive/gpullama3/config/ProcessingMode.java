package org.beehive.gpullama3.config;

/**
 * Defines the processing modes available for prompt and token generation.
 *
 * SERIAL: Traditional token-by-token processing (current implementation)
 * BATCH: llama.cpp-style batch processing for improved context coherence
 */
public enum ProcessingMode {
    /**
     * Process tokens individually in sequence.
     * This is the current implementation that works for all models.
     */
    SERIAL,

    /**
     * Process multiple tokens in batch operations.
     * Designed to solve expert routing context isolation in MoE models like OLMoE.
     */
    BATCH
}