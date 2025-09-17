package org.beehive.gpullama3.isolation.runners;

import org.beehive.gpullama3.isolation.serialization.FloatArraySerializer;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.io.*;

/**
 * Standalone process for executing MLP projector operations in isolation.
 * This prevents TornadoVM GPU deadlocks by running in a separate JVM process.
 * 
 * For now, this is a simple CPU-based implementation that avoids GPU conflicts.
 * The real MLP logic will be implemented once we have proper model loading setup.
 */
public class MLPProjectorProcess {
    
    public static void main(String[] args) {
        String processId = System.getProperty("gpu.process.id", "unknown");
        
        // Add debug file logging
        java.io.PrintWriter debugLog = null;
        try {
            debugLog = new java.io.PrintWriter(new java.io.FileWriter("/tmp/mlp_debug_" + processId + ".log", true));
            debugLog.printf("=== MLP Process Debug Log ===\n");
            debugLog.printf("Process ID: %s\n", processId);
            debugLog.printf("Start time: %s\n", new java.util.Date());
            debugLog.flush();
        } catch (Exception e) {
            // Ignore debug logging failures
        }
        
        System.err.printf("[MLP-PROCESS] Starting MLP projector process: %s%n", processId);
        System.err.printf("[MLP-PROCESS] Max memory: %d MB, Free memory: %d MB%n", 
            Runtime.getRuntime().maxMemory() / (1024*1024),
            Runtime.getRuntime().freeMemory() / (1024*1024));
        
        if (debugLog != null) {
            debugLog.printf("Process started successfully\n");
            debugLog.flush();
        }
        
        long startTime = System.currentTimeMillis();
        
        try {
            // Initialize memory-mapped file serializer  
            org.beehive.gpullama3.isolation.serialization.MemoryMappedFileSerializer mmapSerializer = 
                new org.beehive.gpullama3.isolation.serialization.MemoryMappedFileSerializer(processId);
            
            // Read input from memory-mapped file (file path comes via stdin)
            FloatArray visionTokens;
            try (InputStream stdin = System.in) {
                if (debugLog != null) {
                    debugLog.printf("About to read input file path from stdin\n");
                    debugLog.flush();
                }
                System.err.printf("[MLP-PROCESS] Reading input file path from stdin...%n");
                
                if (debugLog != null) {
                    debugLog.printf("Calling memory-mapped file deserialize...\n");
                    debugLog.flush();
                }
                visionTokens = mmapSerializer.deserialize(stdin);
                if (debugLog != null) {
                    debugLog.printf("Memory-mapped file deserialize completed: %d tokens\n", visionTokens.getSize());
                    debugLog.flush();
                }
                System.err.printf("[MLP-PROCESS] Successfully loaded %d vision tokens from memory-mapped file%n", visionTokens.getSize());
                System.err.printf("[MLP-PROCESS] After deserialization - Free memory: %d MB%n", 
                    Runtime.getRuntime().freeMemory() / (1024*1024));
            } catch (Exception e) {
                if (debugLog != null) {
                    debugLog.printf("Exception during input read: %s\n", e.getMessage());
                    e.printStackTrace(debugLog);
                    debugLog.flush();
                }
                System.err.printf("[MLP-PROCESS] Failed to read input: %s%n", e.getMessage());
                System.err.printf("[MLP-PROCESS] Exception type: %s%n", e.getClass().getName());
                e.printStackTrace();
                throw e;
            }
            
            if (debugLog != null) {
                debugLog.printf("Starting MLP projection...\n");
                debugLog.flush();
            }
            System.err.printf("[MLP-PROCESS] Starting CPU-based MLP projection...%n");
            System.err.printf("[MLP-PROCESS] Before projection - Free memory: %d MB%n", 
                Runtime.getRuntime().freeMemory() / (1024*1024));
            
            // For now, perform a simple CPU-based projection
            // This is a placeholder that transforms 1024-dim to 4096-dim
            FloatArray result = performCPUProjection(visionTokens, debugLog);
            
            if (debugLog != null) {
                debugLog.printf("Projection completed, about to serialize output\n");
                debugLog.flush();
            }
            System.err.printf("[MLP-PROCESS] Projection completed: %d output features%n", result.getSize());
            
            // Write result to memory-mapped file and send file path to stdout
            try (OutputStream stdout = System.out) {
                if (debugLog != null) {
                    debugLog.printf("Serializing result to memory-mapped file...\n");
                    debugLog.flush();
                }
                mmapSerializer.serialize(result, stdout);
                stdout.flush();
                if (debugLog != null) {
                    debugLog.printf("Memory-mapped file serialization completed\n");
                    debugLog.flush();
                }
            }
            
            long duration = System.currentTimeMillis() - startTime;
            if (debugLog != null) {
                debugLog.printf("Process completed successfully in %dms\n", duration);
                debugLog.printf("About to call System.exit(0)\n");
                debugLog.close();
            }
            System.err.printf("[MLP-PROCESS] Process completed successfully in %dms%n", duration);
            
            // Clean exit
            System.exit(0);
            
        } catch (Exception e) {
            long duration = System.currentTimeMillis() - startTime;
            if (debugLog != null) {
                debugLog.printf("Process failed after %dms: %s\n", duration, e.getMessage());
                e.printStackTrace(debugLog);
                debugLog.printf("About to call System.exit(1)\n");
                debugLog.close();
            }
            System.err.printf("[MLP-PROCESS] Process failed after %dms: %s%n", duration, e.getMessage());
            e.printStackTrace();
            
            // Write error to stderr and exit with error code
            System.exit(1);
        }
    }
    
    /**
     * Perform simple CPU-based MLP projection as a placeholder.
     * This transforms vision tokens from 1024-dim CLIP space to 4096-dim LLM space.
     * 
     * In a real implementation, this would:
     * 1. Load actual MLP weights from mmproj file
     * 2. Apply: Linear(1024->4096) + GELU + Linear(4096->4096)
     * 3. Return properly projected embeddings
     */
    private static FloatArray performCPUProjection(FloatArray visionTokens, java.io.PrintWriter debugLog) {
        if (debugLog != null) {
            debugLog.printf("performCPUProjection called\n");
            debugLog.flush();
        }
        System.err.println("[MLP-PROCESS] Performing placeholder CPU projection...");
        
        int inputDim = 1024;   // CLIP feature dimension
        int outputDim = 4096;  // LLM embedding dimension
        
        int numTokens = visionTokens.getSize() / inputDim;
        if (debugLog != null) {
            debugLog.printf("Processing %d tokens (%d -> %d dimensions)\n", numTokens, inputDim, outputDim);
            debugLog.flush();
        }
        System.err.printf("[MLP-PROCESS] Processing %d tokens (%d -> %d dimensions)%n", 
            numTokens, inputDim, outputDim);
        
        // Create output array
        int outputSize = numTokens * outputDim;
        System.err.printf("[MLP-PROCESS] Creating output array: %d elements (%d MB)%n", 
            outputSize, (outputSize * 4) / (1024*1024));
        
        FloatArray result;
        try {
            result = new FloatArray(outputSize);
            System.err.printf("[MLP-PROCESS] Output array created successfully%n");
        } catch (OutOfMemoryError e) {
            System.err.printf("[MLP-PROCESS] OutOfMemoryError creating output array: %s%n", e.getMessage());
            throw new RuntimeException("Insufficient memory for output array", e);
        }
        
        // Simple placeholder transformation: expand and apply basic processing
        for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
            int inputOffset = tokenIdx * inputDim;
            int outputOffset = tokenIdx * outputDim;
            
            // Layer 1: 1024 -> 4096 (simple expansion with processing)
            for (int outDim = 0; outDim < outputDim; outDim++) {
                int inDim = outDim % inputDim; // Cycle through input dimensions
                float inputValue = visionTokens.get(inputOffset + inDim);
                
                // Simple transformation: scale and add bias
                float projectedValue = inputValue * 1.5f + 0.1f;
                
                // Apply simple activation (approximating GELU)
                projectedValue = projectedValue * 0.5f * (1.0f + (float)Math.tanh(projectedValue * 0.7978f));
                
                result.set(outputOffset + outDim, projectedValue);
            }
        }
        
        System.err.printf("[MLP-PROCESS] Placeholder projection completed for %d tokens%n", numTokens);
        return result;
    }
}