package org.beehive.gpullama3.isolation.adapters;

import org.beehive.gpullama3.isolation.core.*;
import org.beehive.gpullama3.isolation.serialization.MemoryMappedFileSerializer;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.time.Duration;

/**
 * Process adapter for MLP Projector GPU operations.
 * Provides isolated execution to prevent TornadoVM GPU deadlocks.
 */
public class MLPProcessAdapter {
    private final ProcessExecutor<FloatArray> processExecutor;
    private final boolean processIsolationEnabled;
    
    public MLPProcessAdapter() {
        this.processIsolationEnabled = getBooleanProperty("llava.mlp.process.isolation.enabled", true);
        this.processExecutor = new ProcessExecutor<>(new MemoryMappedFileSerializer("mlp-process"));
        
        System.err.printf("[MLP-ISOLATION] Initialized: enabled=%s, timeout=PT3M%n", 
            processIsolationEnabled);
    }
    
    /**
     * Execute MLP projection in isolated process to avoid GPU deadlock.
     */
    public FloatArray executeInIsolation(FloatArray visionTokens) throws ProcessExecutionException {
        if (!processIsolationEnabled) {
            throw new ProcessExecutionException("MLP process isolation is disabled");
        }
        
        System.err.printf("[MLP-ISOLATION] Starting isolated MLP projection: %d vision tokens%n", 
            visionTokens.getSize());
        
        try {
            ProcessResult<FloatArray> result = processExecutor.execute(
                "org.beehive.gpullama3.isolation.runners.MLPProjectorProcess",
                visionTokens
            );
            
            System.err.printf("[MLP-ISOLATION] Completed successfully in %dms (attempt %d)%n", 
                result.getDurationMs(), result.getAttemptNumber());
            
            return result.getResult();
            
        } catch (ProcessExecutionException e) {
            System.err.printf("[MLP-ISOLATION] Failed: %s%n", e.getMessage());
            throw e;
        }
    }
    
    /**
     * Execute with custom timeout for specific scenarios.
     */
    public FloatArray executeInIsolation(FloatArray visionTokens, Duration customTimeout) throws ProcessExecutionException {
        if (!processIsolationEnabled) {
            throw new ProcessExecutionException("MLP process isolation is disabled");
        }
        
        try {
            ProcessResult<FloatArray> result = processExecutor.execute(
                "org.beehive.gpullama3.isolation.runners.MLPProjectorProcess",
                visionTokens,
                customTimeout
            );
            
            return result.getResult();
            
        } catch (ProcessExecutionException e) {
            System.err.printf("[MLP-ISOLATION] Failed with custom timeout %s: %s%n", 
                customTimeout, e.getMessage());
            throw e;
        }
    }
    
    /**
     * Check if process isolation is available and properly configured.
     */
    public boolean isAvailable() {
        if (!processIsolationEnabled) {
            return false;
        }
        
        try {
            processExecutor.validateEnvironment();
            return true;
        } catch (ProcessValidationException e) {
            System.err.printf("[MLP-ISOLATION] Environment validation failed: %s%n", e.getMessage());
            return false;
        }
    }
    
    /**
     * Check if process isolation is enabled.
     */
    public boolean isEnabled() {
        return processIsolationEnabled;
    }
    
    /**
     * Validate environment and log any issues.
     */
    public void validateEnvironment() throws ProcessValidationException {
        processExecutor.validateEnvironment();
        System.err.println("[MLP-ISOLATION] Environment validation passed");
    }
    
    private boolean getBooleanProperty(String propertyName, boolean defaultValue) {
        String property = System.getProperty(propertyName);
        if (property == null) {
            return defaultValue;
        }
        return "true".equalsIgnoreCase(property);
    }
    
    @Override
    public String toString() {
        return String.format("MLPProcessAdapter{enabled=%s}", 
            processIsolationEnabled);
    }
}