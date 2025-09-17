package org.beehive.gpullama3.isolation.core;

import java.time.Duration;

/**
 * Result wrapper for subprocess execution containing both the result and performance metrics.
 */
public class ProcessResult<T> {
    private final T result;
    private final Duration duration;
    private final String processId;
    private final int attemptNumber;
    private final long memoryUsedBytes;
    
    public ProcessResult(T result, Duration duration, String processId, int attemptNumber) {
        this(result, duration, processId, attemptNumber, -1);
    }
    
    public ProcessResult(T result, Duration duration, String processId, int attemptNumber, long memoryUsedBytes) {
        this.result = result;
        this.duration = duration;
        this.processId = processId;
        this.attemptNumber = attemptNumber;
        this.memoryUsedBytes = memoryUsedBytes;
    }
    
    /**
     * Get the actual result from the subprocess execution.
     */
    public T getResult() {
        return result;
    }
    
    /**
     * Get the total execution duration.
     */
    public Duration getDuration() {
        return duration;
    }
    
    /**
     * Get execution duration in milliseconds for logging.
     */
    public long getDurationMs() {
        return duration.toMillis();
    }
    
    /**
     * Get the unique process identifier.
     */
    public String getProcessId() {
        return processId;
    }
    
    /**
     * Get the attempt number (1-based) that succeeded.
     */
    public int getAttemptNumber() {
        return attemptNumber;
    }
    
    /**
     * Get memory usage if available, -1 if not measured.
     */
    public long getMemoryUsedBytes() {
        return memoryUsedBytes;
    }
    
    /**
     * Check if this was a retry (attempt > 1).
     */
    public boolean wasRetried() {
        return attemptNumber > 1;
    }
    
    @Override
    public String toString() {
        return String.format("ProcessResult{processId='%s', attempt=%d, duration=%dms, memoryMB=%.1f}", 
            processId, attemptNumber, getDurationMs(), memoryUsedBytes / (1024.0 * 1024.0));
    }
}