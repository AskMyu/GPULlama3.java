package org.beehive.gpullama3.isolation.core;

import java.util.List;
import java.util.Map;

/**
 * Core abstraction for GPU components that can be executed in isolated processes.
 * Provides type-safe interface for input/output and environment configuration.
 */
public interface GPUProcessComponent<I, O> {
    
    /**
     * Execute the GPU component with the given input.
     * This method runs in the isolated subprocess.
     */
    O execute(I input) throws ProcessExecutionException;
    
    /**
     * Get the component name for logging and identification.
     */
    String getComponentName();
    
    /**
     * Get additional environment variables required by this component.
     */
    default Map<String, String> getEnvironmentVariables() {
        return Map.of();
    }
    
    /**
     * Get additional JVM arguments required by this component.
     */
    default List<String> getJVMArguments() {
        return List.of();
    }
    
    /**
     * Initialize the component in the subprocess environment.
     * Called once before execute().
     */
    default void initialize() throws ProcessExecutionException {
        // Default: no initialization required
    }
    
    /**
     * Clean up resources after execution.
     * Called once after execute() completes.
     */
    default void cleanup() {
        // Default: no cleanup required
    }
    
    /**
     * Validate that this component can run in the current environment.
     */
    default void validateEnvironment() throws ProcessValidationException {
        // Default: no validation required
    }
}