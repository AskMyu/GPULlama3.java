package org.beehive.gpullama3.process;

import java.io.*;

/**
 * ProcessWorker - Main class for subprocess execution.
 * 
 * This class runs in the isolated JVM subprocess and executes GPU operations
 * while maintaining complete GPU context isolation from the parent process.
 */
public class ProcessWorker {
    
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: ProcessWorker <operationClass> <inputFile> <tempDir>");
            System.exit(1);
        }
        
        String operationClassName = args[0];
        String inputFilePath = args[1];
        String tempDirPath = args[2];
        
        System.out.println("[ProcessWorker] Starting GPU process isolation worker");
        System.out.println("[ProcessWorker] Operation: " + operationClassName);
        System.out.println("[ProcessWorker] Input: " + inputFilePath);
        System.out.println("[ProcessWorker] TempDir: " + tempDirPath);
        
        try {
            // Load operation class
            Class<?> operationClass = Class.forName(operationClassName);
            ProcessOperation<?> operation = (ProcessOperation<?>) operationClass.getDeclaredConstructor().newInstance();
            
            // Read input data
            File inputFile = new File(inputFilePath);
            Serializable input = SerializationUtils.deserialize(inputFile, Serializable.class);
            
            System.out.println("[ProcessWorker] Loaded operation: " + operation.getOperationName());
            System.out.println("[ProcessWorker] Input data size: " + SerializationUtils.getSerializedSize(input) + " bytes");
            
            // Execute operation in isolated GPU context
            System.out.println("[ProcessWorker] ===== STARTING GPU OPERATION =====");
            long startTime = System.currentTimeMillis();
            
            Object result = operation.execute(input);
            
            long executionTime = System.currentTimeMillis() - startTime;
            System.out.println("[ProcessWorker] ===== GPU OPERATION COMPLETED =====");
            System.out.println("[ProcessWorker] Execution time: " + executionTime + "ms");
            
            // Write result
            File outputFile = new File(tempDirPath, "output.ser");
            SerializationUtils.serialize((Serializable) result, outputFile);
            
            System.out.println("[ProcessWorker] Result written to: " + outputFile.getAbsolutePath());
            System.out.println("[ProcessWorker] Process completed successfully");
            
            // Clean shutdown to release all GPU resources
            System.exit(0);
            
        } catch (Exception e) {
            System.err.println("[ProcessWorker] FATAL ERROR: " + e.getMessage());
            e.printStackTrace();
            
            // Write error information
            try {
                File errorFile = new File(tempDirPath, "error.txt");
                try (PrintWriter writer = new PrintWriter(errorFile)) {
                    writer.println("Error: " + e.getMessage());
                    writer.println("Class: " + e.getClass().getName());
                    e.printStackTrace(writer);
                }
            } catch (IOException ioError) {
                System.err.println("[ProcessWorker] Failed to write error file: " + ioError.getMessage());
            }
            
            System.exit(1);
        }
    }
}