package org.beehive.gpullama3.isolation.runners;

import java.io.*;

/**
 * Simple test process to verify subprocess execution works.
 */
public class SimpleTestProcess {
    
    public static void main(String[] args) {
        String processId = System.getProperty("gpu.process.id", "unknown");
        System.err.printf("[TEST-PROCESS] Starting test process: %s%n", processId);
        System.err.printf("[TEST-PROCESS] Java version: %s%n", System.getProperty("java.version"));
        System.err.printf("[TEST-PROCESS] Max memory: %d MB, Free memory: %d MB%n", 
            Runtime.getRuntime().maxMemory() / (1024*1024),
            Runtime.getRuntime().freeMemory() / (1024*1024));
        
        try {
            // Read from stdin (expecting simple text)
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
                System.err.printf("[TEST-PROCESS] Reading from stdin...%n");
                String line = reader.readLine();
                System.err.printf("[TEST-PROCESS] Read: %s%n", line);
                
                // Write back to stdout
                System.out.println("HELLO FROM SUBPROCESS: " + line);
                System.out.flush();
                
                System.err.printf("[TEST-PROCESS] Process completed successfully%n");
                System.exit(0);
            }
            
        } catch (Exception e) {
            System.err.printf("[TEST-PROCESS] Process failed: %s%n", e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}