package org.beehive.gpullama3.process;

import java.io.Serializable;

/**
 * Interface for operations that can be executed in isolated GPU processes.
 * All implementations must be serializable for IPC communication.
 */
public interface ProcessOperation<T> extends Serializable {
    
    /**
     * Get the input data for this operation.
     * Input must be serializable for process communication.
     */
    Serializable getInput();
    
    /**
     * Get the expected output class for result deserialization.
     */
    Class<T> getOutputClass();
    
    /**
     * Execute the operation with the provided input.
     * This method runs in the isolated subprocess.
     */
    T execute(Serializable input) throws Exception;
    
    /**
     * Get operation name for logging and monitoring.
     */
    default String getOperationName() {
        return this.getClass().getSimpleName();
    }
    
    /**
     * Get estimated execution time in seconds for timeout management.
     */
    default int getEstimatedTimeoutSeconds() {
        return 60; // Default 60 seconds
    }
}