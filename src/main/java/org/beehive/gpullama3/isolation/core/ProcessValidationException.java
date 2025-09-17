package org.beehive.gpullama3.isolation.core;

/**
 * Exception thrown when process environment validation fails.
 */
public class ProcessValidationException extends ProcessExecutionException {
    public ProcessValidationException(String message) {
        super("Environment validation failed: " + message);
    }
    
    public ProcessValidationException(String message, Throwable cause) {
        super("Environment validation failed: " + message, cause);
    }
}