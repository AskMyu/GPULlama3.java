package org.beehive.gpullama3.isolation.core;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Core process execution engine for GPU component isolation.
 * Provides generic subprocess management with comprehensive error handling and retry logic.
 */
public class ProcessExecutor<T> {
    private static final AtomicLong PROCESS_COUNTER = new AtomicLong(0);
    private static final int MAX_RETRIES = 3;
    private static final Duration DEFAULT_TIMEOUT = Duration.ofMinutes(2);
    
    private final int maxRetries;
    private final ProcessSerializer<T> serializer;
    private final String processId;
    
    public ProcessExecutor(ProcessSerializer<T> serializer) {
        this.maxRetries = MAX_RETRIES;
        this.serializer = serializer;
        this.processId = "gpu-process-" + PROCESS_COUNTER.incrementAndGet();
    }
    
    /**
     * Execute a GPU component in an isolated subprocess with retry logic.
     */
    public ProcessResult<T> execute(String processClassName, T input) throws ProcessExecutionException {
        return execute(processClassName, input, DEFAULT_TIMEOUT);
    }
    
    /**
     * Execute a GPU component in an isolated subprocess with custom timeout.
     */
    public ProcessResult<T> execute(String processClassName, T input, Duration timeout) throws ProcessExecutionException {
        validateEnvironment();
        
        ProcessExecutionException lastException = null;
        
        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                System.err.printf("[GPU-ISOLATION] Attempt %d/%d: Starting %s (processId=%s)%n", 
                    attempt, maxRetries, processClassName, processId);
                
                ProcessResult<T> result = executeOnce(processClassName, input, timeout, attempt);
                
                System.err.printf("[GPU-ISOLATION] Success: %s completed in %dms%n", 
                    processClassName, result.getDurationMs());
                
                return result;
                
            } catch (ProcessExecutionException e) {
                lastException = e;
                
                if (attempt == maxRetries || !isRetryableError(e)) {
                    System.err.printf("[GPU-ISOLATION] Fatal error on attempt %d: %s%n", attempt, e.getMessage());
                    break;
                }
                
                System.err.printf("[GPU-ISOLATION] Retry attempt %d failed: %s%n", attempt, e.getMessage());
                
                try {
                    // Exponential backoff
                    long waitMs = 1000 * attempt;
                    Thread.sleep(waitMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new ProcessExecutionException("Interrupted during retry backoff", ie);
                }
            }
        }
        
        throw new ProcessExecutionException(
            String.format("Process execution failed after %d attempts", maxRetries), 
            lastException
        );
    }
    
    /**
     * Single execution attempt without retry logic.
     */
    private ProcessResult<T> executeOnce(String processClassName, T input, Duration timeout, int attempt) 
            throws ProcessExecutionException {
        
        Instant startTime = Instant.now();
        Process process = null;
        
        try {
            // Build process command
            List<String> command = buildProcessCommand(processClassName, attempt);
            
            // Start subprocess
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            // Environment variables are inherited automatically from parent process
            
            // Explicitly pass critical TornadoVM environment variables
            Map<String, String> env = processBuilder.environment();
            if (System.getenv("TORNADO_ROOT") != null) {
                env.put("TORNADO_ROOT", System.getenv("TORNADO_ROOT"));
            }
            if (System.getenv("TORNADO_SDK") != null) {
                env.put("TORNADO_SDK", System.getenv("TORNADO_SDK"));
            }
            if (System.getenv("TORNADO_JAR_DIR") != null) {
                env.put("TORNADO_JAR_DIR", System.getenv("TORNADO_JAR_DIR"));
                System.err.printf("[GPU-ISOLATION] Passing TORNADO_JAR_DIR to subprocess: %s%n", 
                    System.getenv("TORNADO_JAR_DIR"));
            }
            
            process = processBuilder.start();
            
            // Memory-mapped file communication setup
            System.err.printf("[GPU-ISOLATION] Using memory-mapped file communication%n");
            
            // Send input via memory-mapped file
            try (OutputStream processInput = process.getOutputStream()) {
                System.err.printf("[GPU-ISOLATION] Serializing input data to memory-mapped file...%n");
                serializer.serialize(input, processInput);
                processInput.flush();
                System.err.printf("[GPU-ISOLATION] Input data written to memory-mapped file successfully%n");
            } catch (ProcessSerializer.SerializationException e) {
                System.err.printf("[GPU-ISOLATION] Serialization failed: %s%n", e.getMessage());
                
                // Try to read stderr to see why subprocess died
                try {
                    String stderr = readStream(process.getErrorStream());
                    if (!stderr.isEmpty()) {
                        System.err.printf("[GPU-ISOLATION] Subprocess stderr: %s%n", stderr);
                    } else {
                        System.err.printf("[GPU-ISOLATION] No stderr output from subprocess%n");
                    }
                } catch (Exception readEx) {
                    System.err.printf("[GPU-ISOLATION] Could not read subprocess stderr: %s%n", readEx.getMessage());
                }
                
                throw new ProcessExecutionException("Failed to serialize input", e);
            }
            
            // Read subprocess output from memory-mapped file
            final Process finalProcess = process; // Make final for lambda
            java.util.concurrent.Future<T> outputReader = java.util.concurrent.Executors.newSingleThreadExecutor().submit(() -> {
                try (InputStream processOutput = finalProcess.getInputStream()) {
                    System.err.printf("[GPU-ISOLATION] Reading subprocess output from memory-mapped file...%n");
                    
                    T result = serializer.deserialize(processOutput);
                    System.err.printf("[GPU-ISOLATION] Memory-mapped file deserialization completed successfully%n");
                    return result;
                } catch (Exception e) {
                    System.err.printf("[GPU-ISOLATION] Memory-mapped file reader failed: %s%n", e.getMessage());
                    e.printStackTrace(System.err);
                    throw new RuntimeException("Failed to read subprocess output from memory-mapped file", e);
                }
            });
            
            // Wait for completion with timeout
            boolean completed = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS);
            
            if (!completed) {
                process.destroyForcibly();
                throw new ProcessExecutionException("Process timed out after " + timeout);
            }
            
            // Check exit code
            int exitCode = process.exitValue();
            if (exitCode != 0) {
                String stderr = readStream(process.getErrorStream());
                throw new ProcessExecutionException(
                    String.format("Process exited with code %d: %s", exitCode, stderr));
            }
            
            // Get the deserialized result from the concurrent reader
            T result;
            try {
                System.err.printf("[GPU-ISOLATION] Waiting for subprocess result from concurrent reader...%n");
                result = outputReader.get(30, TimeUnit.SECONDS); // Increased timeout for debugging
                System.err.printf("[GPU-ISOLATION] Successfully got subprocess result from concurrent reader%n");
            } catch (java.util.concurrent.TimeoutException e) {
                System.err.printf("[GPU-ISOLATION] Timeout waiting for subprocess result: %s%n", e.getMessage());
                System.err.printf("[GPU-ISOLATION] Process exit code: %d%n", process.exitValue());
                // Cancel the reader task
                outputReader.cancel(true);
                throw new ProcessExecutionException("Timeout waiting for subprocess result", e);
            } catch (Exception e) {
                System.err.printf("[GPU-ISOLATION] Failed to get subprocess result: %s%n", e.getMessage());
                System.err.printf("[GPU-ISOLATION] Exception type: %s%n", e.getClass().getSimpleName());
                
                // Try to read stderr for additional error information
                try {
                    String stderr = readStream(process.getErrorStream());
                    if (!stderr.isEmpty()) {
                        System.err.printf("[GPU-ISOLATION] Subprocess stderr: %s%n", stderr);
                    }
                } catch (IOException ioEx) {
                    System.err.printf("[GPU-ISOLATION] Could not read subprocess stderr: %s%n", ioEx.getMessage());
                }
                
                throw new ProcessExecutionException("Failed to get subprocess result", e);
            }
            
            Duration duration = Duration.between(startTime, Instant.now());
            
            return new ProcessResult<>(result, duration, processId, attempt);
            
        } catch (IOException e) {
            throw new ProcessExecutionException("IO error during process execution", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ProcessExecutionException("Process execution interrupted", e);
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
            }
            
            // Clean up memory-mapped files
            if (serializer instanceof org.beehive.gpullama3.isolation.serialization.MemoryMappedFileSerializer) {
                ((org.beehive.gpullama3.isolation.serialization.MemoryMappedFileSerializer) serializer).cleanup();
            }
        }
    }
    
    /**
     * Build the subprocess command with proper JVM arguments and classpath.
     */
    private List<String> buildProcessCommand(String processClassName, int attempt) {
        List<String> command = new ArrayList<>();
        
        // Java executable
        String javaHome = System.getenv("JAVA_HOME");
        if (javaHome != null) {
            command.add(javaHome + "/bin/java");
        } else {
            command.add("java");
        }
        
        // JVM arguments - add basic required arguments
        command.add("--enable-preview");
        command.add("--add-modules");
        command.add("jdk.incubator.vector");
        
        // Add memory settings for subprocess (ensure enough memory for large FloatArrays)
        command.add("-Xmx2g");  // Max heap 2GB
        command.add("-Xms512m"); // Initial heap 512MB
        
        // Classpath - make sure to include current classpath
        String classpath = System.getProperty("java.class.path");
        System.err.printf("[GPU-ISOLATION] DEBUG: Original classpath contains tornado: %s%n", 
            classpath.contains("tornado"));
        System.err.printf("[GPU-ISOLATION] DEBUG: Classpath entries: %s%n", 
            java.util.Arrays.toString(classpath.split(":")));
        
        // Add TornadoVM JARs from environment-specified directory
        String tornadoJarDir = System.getenv("TORNADO_JAR_DIR");
        if (tornadoJarDir != null) {
            java.io.File jarDir = new java.io.File(tornadoJarDir);
            if (jarDir.exists() && jarDir.isDirectory()) {
                java.io.File[] jarFiles = jarDir.listFiles(file -> 
                    file.getName().startsWith("tornado-") && 
                    file.getName().endsWith(".jar") && 
                    !file.getName().contains("-sources"));
                if (jarFiles != null) {
                    for (java.io.File jarFile : jarFiles) {
                        classpath = classpath + ":" + jarFile.getAbsolutePath();
                    }
                    System.err.printf("[GPU-ISOLATION] Added %d TornadoVM JARs from %s to classpath%n", 
                        jarFiles.length, tornadoJarDir);
                } else {
                    System.err.printf("[GPU-ISOLATION] Warning: No TornadoVM JARs found in %s%n", tornadoJarDir);
                }
            } else {
                System.err.printf("[GPU-ISOLATION] Warning: TornadoVM JAR directory not found: %s%n", tornadoJarDir);
            }
        } else {
            System.err.printf("[GPU-ISOLATION] Warning: TORNADO_JAR_DIR environment variable not set%n");
        }
        
        command.add("-cp");
        command.add(classpath);
        
        System.err.printf("[GPU-ISOLATION] Using classpath (%d chars): %s%n", classpath.length(), 
            classpath.length() > 200 ? classpath.substring(0, 200) + "..." : classpath);
        
        // Process identification
        command.add("-Dgpu.process.id=" + processId + "-" + attempt);
        command.add("-Dgpu.process.isolation=true");
        
        // Main class
        command.add(processClassName);
        
        return command;
    }
    
    /**
     * Validate that the environment is properly configured for TornadoVM execution.
     */
    public void validateEnvironment() throws ProcessValidationException {
        // Check Java version
        String javaVersion = System.getProperty("java.version");
        if (!javaVersion.startsWith("21")) {
            System.err.printf("[GPU-ISOLATION] Warning: Java %s detected, Java 21+ recommended%n", javaVersion);
        }
        
        // Validate JAVA_HOME if specified
        String javaHome = System.getenv("JAVA_HOME");
        if (javaHome != null) {
            File javaBin = new File(javaHome, "bin/java");
            if (!javaBin.exists()) {
                throw new ProcessValidationException("Invalid JAVA_HOME: " + javaHome);
            }
        }
    }
    
    /**
     * Determine if an exception indicates a retryable error condition.
     */
    private boolean isRetryableError(ProcessExecutionException e) {
        String message = e.getMessage().toLowerCase();
        
        // GPU-related transient errors that may be retryable
        return message.contains("gpu") && 
               (message.contains("timeout") || 
                message.contains("device busy") || 
                message.contains("out of memory") ||
                message.contains("kernel launch"));
    }
    
    /**
     * Read entire stream to string for error reporting.
     */
    private String readStream(InputStream stream) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString().trim();
    }
    
    public String getProcessId() {
        return processId;
    }
}