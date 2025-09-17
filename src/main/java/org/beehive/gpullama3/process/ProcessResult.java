package org.beehive.gpullama3.process;

/**
 * Result wrapper for process execution containing the output data
 * and performance metrics.
 */
public class ProcessResult<T> {
    
    private final T result;
    private final long executionTimeMs;
    private final int attemptNumber;
    private final long timestamp;
    
    public ProcessResult(T result, long executionTimeMs, int attemptNumber) {
        this.result = result;
        this.executionTimeMs = executionTimeMs;
        this.attemptNumber = attemptNumber;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Get the operation result.
     */
    public T getResult() {
        return result;
    }
    
    /**
     * Get execution time in milliseconds.
     */
    public long getExecutionTimeMs() {
        return executionTimeMs;
    }
    
    /**
     * Get execution time in seconds.
     */
    public double getExecutionTimeSeconds() {
        return executionTimeMs / 1000.0;
    }
    
    /**
     * Get the attempt number (1-based) that succeeded.
     */
    public int getAttemptNumber() {
        return attemptNumber;
    }
    
    /**
     * Get timestamp when result was created.
     */
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * Check if this was the first attempt (no retries needed).
     */
    public boolean isFirstAttempt() {
        return attemptNumber == 1;
    }
    
    @Override
    public String toString() {
        return String.format("ProcessResult{attempt=%d, time=%.2fs, result=%s}", 
            attemptNumber, getExecutionTimeSeconds(), 
            result != null ? result.getClass().getSimpleName() : "null");
    }
}