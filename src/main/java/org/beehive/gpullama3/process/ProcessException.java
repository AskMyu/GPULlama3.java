package org.beehive.gpullama3.process;

/**
 * Exception thrown when process execution fails.
 * Provides specific error handling for GPU process isolation scenarios.
 */
public class ProcessException extends RuntimeException {
    
    private final String processId;
    private final int attemptNumber;
    
    public ProcessException(String message) {
        super(message);
        this.processId = null;
        this.attemptNumber = 0;
    }
    
    public ProcessException(String message, Throwable cause) {
        super(message, cause);
        this.processId = null;
        this.attemptNumber = 0;
    }
    
    public ProcessException(String message, String processId, int attemptNumber) {
        super(message);
        this.processId = processId;
        this.attemptNumber = attemptNumber;
    }
    
    public ProcessException(String message, Throwable cause, String processId, int attemptNumber) {
        super(message, cause);
        this.processId = processId;
        this.attemptNumber = attemptNumber;
    }
    
    /**
     * Get the process ID that failed (if available).
     */
    public String getProcessId() {
        return processId;
    }
    
    /**
     * Get the attempt number that failed.
     */
    public int getAttemptNumber() {
        return attemptNumber;
    }
    
    /**
     * Check if this was a timeout-related failure.
     */
    public boolean isTimeout() {
        String message = getMessage();
        return message != null && message.toLowerCase().contains("timeout");
    }
    
    /**
     * Check if this was a GPU-related failure.
     */
    public boolean isGpuRelated() {
        String message = getMessage();
        if (message == null) return false;
        
        String lowerMessage = message.toLowerCase();
        return lowerMessage.contains("gpu") || 
               lowerMessage.contains("tornado") || 
               lowerMessage.contains("cuda") ||
               lowerMessage.contains("opencl");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("ProcessException");
        if (processId != null) {
            sb.append(" [").append(processId).append("]");
        }
        if (attemptNumber > 0) {
            sb.append(" (attempt ").append(attemptNumber).append(")");
        }
        sb.append(": ").append(getMessage());
        return sb.toString();
    }
}