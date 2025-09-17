package org.beehive.gpullama3.isolation.core;

/**
 * Exception thrown when subprocess execution fails.
 */
public class ProcessExecutionException extends Exception {
    private final String processId;
    private final int attemptNumber;
    
    public ProcessExecutionException(String message) {
        this(message, null, null, -1);
    }
    
    public ProcessExecutionException(String message, Throwable cause) {
        this(message, cause, null, -1);
    }
    
    public ProcessExecutionException(String message, Throwable cause, String processId, int attemptNumber) {
        super(message, cause);
        this.processId = processId;
        this.attemptNumber = attemptNumber;
    }
    
    public String getProcessId() {
        return processId;
    }
    
    public int getAttemptNumber() {
        return attemptNumber;
    }
    
    @Override
    public String getMessage() {
        String baseMessage = super.getMessage();
        if (processId != null) {
            return String.format("%s (processId=%s, attempt=%d)", baseMessage, processId, attemptNumber);
        }
        return baseMessage;
    }
}

