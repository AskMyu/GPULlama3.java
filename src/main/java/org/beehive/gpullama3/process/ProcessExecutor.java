package org.beehive.gpullama3.process;

import java.io.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.List;
import java.util.ArrayList;

/**
 * Core JVM Process Isolation Framework - ProcessExecutor
 * 
 * Manages subprocess execution for GPU-intensive operations to resolve
 * TornadoVM GPU context deadlocks. Provides generic, reusable process
 * isolation that can be easily plugged in/out of different architectures.
 */
public class ProcessExecutor implements AutoCloseable {
    
    private static final int DEFAULT_TIMEOUT_SECONDS = 120;
    private static final int MAX_RETRIES = 3;
    private static final AtomicInteger processIdCounter = new AtomicInteger(0);
    
    private final String processId;
    private final ExecutorService executorService;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    
    // Process lifecycle management
    private Process currentProcess;
    private Future<?> monitorTask;
    
    public ProcessExecutor() {
        this.processId = "proc-" + processIdCounter.incrementAndGet();
        this.executorService = Executors.newFixedThreadPool(3, r -> {
            Thread t = new Thread(r, "ProcessExecutor-" + processId);
            t.setDaemon(true);
            return t;
        });
        
        System.out.println("[ProcessExecutor-" + processId + "] Initialized GPU process isolation framework");
    }
    
    /**
     * Execute a GPU operation in an isolated JVM process.
     * 
     * @param operation The operation to execute
     * @param timeoutSeconds Maximum execution time
     * @return ProcessResult containing output and performance metrics
     */
    public <T> ProcessResult<T> execute(ProcessOperation<T> operation, int timeoutSeconds) {
        if (closed.get()) {
            throw new IllegalStateException("ProcessExecutor is closed");
        }
        
        System.out.println("[ProcessExecutor-" + processId + "] Starting isolated GPU process execution");
        long startTime = System.currentTimeMillis();
        
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                return executeWithRetry(operation, timeoutSeconds, attempt);
            } catch (ProcessException e) {
                System.err.println("[ProcessExecutor-" + processId + "] Attempt " + attempt + " failed: " + e.getMessage());
                if (attempt == MAX_RETRIES) {
                    throw e;
                }
                // Brief delay before retry
                try {
                    Thread.sleep(1000 * attempt);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new ProcessException("Interrupted during retry delay", ie);
                }
            }
        }
        
        throw new ProcessException("All retry attempts failed");
    }
    
    /**
     * Execute with default timeout.
     */
    public <T> ProcessResult<T> execute(ProcessOperation<T> operation) {
        return execute(operation, DEFAULT_TIMEOUT_SECONDS);
    }
    
    private <T> ProcessResult<T> executeWithRetry(ProcessOperation<T> operation, int timeoutSeconds, int attempt) {
        File tempDir = null;
        try {
            // Create temporary directory for IPC
            tempDir = createTempDirectory();
            
            // Serialize operation input
            File inputFile = new File(tempDir, "input.ser");
            SerializationUtils.serialize(operation.getInput(), inputFile);
            
            // Build subprocess command
            List<String> command = buildSubprocessCommand(operation, inputFile, tempDir);
            
            // Launch subprocess
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.directory(new File(System.getProperty("user.dir")));
            pb.redirectErrorStream(true);
            
            System.out.println("[ProcessExecutor-" + processId + "] Launching subprocess: " + String.join(" ", command));
            
            currentProcess = pb.start();
            
            // Monitor process execution
            ProcessMonitor monitor = new ProcessMonitor(currentProcess, timeoutSeconds);
            monitorTask = executorService.submit(monitor);
            
            // Wait for completion
            boolean completed = currentProcess.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            
            if (!completed) {
                killProcess();
                throw new ProcessException("Process timeout after " + timeoutSeconds + " seconds");
            }
            
            int exitCode = currentProcess.exitValue();
            if (exitCode != 0) {
                String output = readProcessOutput(currentProcess);
                throw new ProcessException("Process failed with exit code " + exitCode + ": " + output);
            }
            
            // Read results
            File outputFile = new File(tempDir, "output.ser");
            if (!outputFile.exists()) {
                throw new ProcessException("Process completed but output file not found");
            }
            
            T result = SerializationUtils.deserialize(outputFile, operation.getOutputClass());
            
            long executionTime = System.currentTimeMillis() - System.currentTimeMillis();
            System.out.println("[ProcessExecutor-" + processId + "] Process completed successfully in " + executionTime + "ms");
            
            return new ProcessResult<>(result, executionTime, attempt);
            
        } catch (Exception e) {
            killProcess();
            throw new ProcessException("Process execution failed on attempt " + attempt, e);
        } finally {
            cleanup(tempDir);
        }
    }
    
    private List<String> buildSubprocessCommand(ProcessOperation<?> operation, File inputFile, File tempDir) {
        List<String> command = new ArrayList<>();
        
        // Java executable
        String javaHome = System.getProperty("java.home");
        command.add(javaHome + "/bin/java");
        
        // JVM arguments for TornadoVM
        command.add("--enable-preview");
        command.add("--add-modules");
        command.add("jdk.incubator.vector");
        command.add("-Xmx8g");
        command.add("-XX:+UseG1GC");
        
        // TornadoVM environment
        String tornadoRoot = System.getenv("TORNADO_ROOT");
        if (tornadoRoot != null) {
            command.add("-Dtornado.root=" + tornadoRoot);
        }
        
        // Classpath
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        
        // Main class for subprocess execution
        command.add("org.beehive.gpullama3.process.ProcessWorker");
        
        // Operation parameters
        command.add(operation.getClass().getName());
        command.add(inputFile.getAbsolutePath());
        command.add(tempDir.getAbsolutePath());
        
        return command;
    }
    
    private File createTempDirectory() throws IOException {
        File tempDir = File.createTempFile("gpu-process-", "");
        tempDir.delete();
        tempDir.mkdirs();
        return tempDir;
    }
    
    private void killProcess() {
        if (currentProcess != null && currentProcess.isAlive()) {
            System.out.println("[ProcessExecutor-" + processId + "] Terminating subprocess");
            currentProcess.destroyForcibly();
        }
        if (monitorTask != null) {
            monitorTask.cancel(true);
        }
    }
    
    private String readProcessOutput(Process process) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            return reader.lines().reduce("", (a, b) -> a + "\n" + b);
        } catch (IOException e) {
            return "Failed to read process output: " + e.getMessage();
        }
    }
    
    private void cleanup(File tempDir) {
        if (tempDir != null && tempDir.exists()) {
            deleteDirectory(tempDir);
        }
    }
    
    private void deleteDirectory(File dir) {
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    deleteDirectory(file);
                } else {
                    file.delete();
                }
            }
        }
        dir.delete();
    }
    
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            System.out.println("[ProcessExecutor-" + processId + "] Shutting down process executor");
            killProcess();
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
    
    /**
     * Process monitor for tracking subprocess health
     */
    private static class ProcessMonitor implements Runnable {
        private final Process process;
        private final int timeoutSeconds;
        
        public ProcessMonitor(Process process, int timeoutSeconds) {
            this.process = process;
            this.timeoutSeconds = timeoutSeconds;
        }
        
        @Override
        public void run() {
            try {
                // Monitor process periodically
                for (int i = 0; i < timeoutSeconds; i++) {
                    if (!process.isAlive()) {
                        return; // Process completed
                    }
                    Thread.sleep(1000);
                }
                
                // Timeout reached
                if (process.isAlive()) {
                    System.err.println("[ProcessMonitor] Process timeout reached, terminating");
                    process.destroyForcibly();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}