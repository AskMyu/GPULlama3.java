package org.beehive.gpullama3.isolation.runners;

import org.beehive.gpullama3.isolation.core.GPUProcessComponent;
import org.beehive.gpullama3.isolation.core.ProcessExecutionException;

import java.io.*;
import java.lang.reflect.Constructor;

/**
 * Generic process runner for GPU components implementing GPUProcessComponent interface.
 * Provides a reusable execution pattern for any GPU component that needs process isolation.
 */
public class GenericGPUComponentProcess {
    
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("[GPU-COMPONENT-PROCESS] Error: Component class name required as first argument");
            System.exit(1);
        }
        
        String componentClassName = args[0];
        String processId = System.getProperty("gpu.process.id", "unknown");
        
        System.err.printf("[GPU-COMPONENT-PROCESS] Starting: %s (processId=%s)%n", 
            componentClassName, processId);
        
        long startTime = System.currentTimeMillis();
        GPUProcessComponent<?, ?> component = null;
        
        try {
            // Instantiate the component
            component = instantiateComponent(componentClassName);
            
            System.err.printf("[GPU-COMPONENT-PROCESS] Component instantiated: %s%n", 
                component.getComponentName());
            
            // Execute the component
            executeComponent(component);
            
            long duration = System.currentTimeMillis() - startTime;
            System.err.printf("[GPU-COMPONENT-PROCESS] Completed successfully in %dms%n", duration);
            
            System.exit(0);
            
        } catch (Exception e) {
            long duration = System.currentTimeMillis() - startTime;
            System.err.printf("[GPU-COMPONENT-PROCESS] Failed after %dms: %s%n", duration, e.getMessage());
            e.printStackTrace();
            
            System.exit(1);
            
        } finally {
            // Cleanup
            if (component != null) {
                try {
                    component.cleanup();
                    System.err.printf("[GPU-COMPONENT-PROCESS] Component cleanup completed%n");
                } catch (Exception e) {
                    System.err.printf("[GPU-COMPONENT-PROCESS] Warning: Cleanup failed: %s%n", e.getMessage());
                }
            }
        }
    }
    
    /**
     * Instantiate the GPU component using reflection.
     */
    @SuppressWarnings("unchecked")
    private static GPUProcessComponent<?, ?> instantiateComponent(String className) throws Exception {
        System.err.printf("[GPU-COMPONENT-PROCESS] Loading component class: %s%n", className);
        
        try {
            Class<?> componentClass = Class.forName(className);
            
            if (!GPUProcessComponent.class.isAssignableFrom(componentClass)) {
                throw new IllegalArgumentException(
                    "Component class must implement GPUProcessComponent: " + className);
            }
            
            // Try to instantiate with default constructor
            Constructor<?> constructor = componentClass.getDeclaredConstructor();
            return (GPUProcessComponent<?, ?>) constructor.newInstance();
            
        } catch (ClassNotFoundException e) {
            throw new Exception("Component class not found: " + className, e);
        } catch (NoSuchMethodException e) {
            throw new Exception("Component class must have a default constructor: " + className, e);
        } catch (Exception e) {
            throw new Exception("Failed to instantiate component: " + className, e);
        }
    }
    
    /**
     * Execute the component with generic input/output handling.
     * This uses Object serialization for now - specific implementations should use typed serializers.
     */
    @SuppressWarnings("unchecked")
    private static void executeComponent(GPUProcessComponent component) throws Exception {
        // Validate environment
        component.validateEnvironment();
        
        // Initialize component
        component.initialize();
        
        System.err.printf("[GPU-COMPONENT-PROCESS] Reading input from stdin%n");
        
        // Read input using Java object serialization (fallback approach)
        Object input;
        try (ObjectInputStream ois = new ObjectInputStream(System.in)) {
            input = ois.readObject();
            System.err.printf("[GPU-COMPONENT-PROCESS] Input received: %s%n", input.getClass().getSimpleName());
        }
        
        // Execute component
        System.err.printf("[GPU-COMPONENT-PROCESS] Executing component: %s%n", component.getComponentName());
        Object result = component.execute(input);
        
        System.err.printf("[GPU-COMPONENT-PROCESS] Execution completed: %s%n", result.getClass().getSimpleName());
        
        // Write output using Java object serialization
        try (ObjectOutputStream oos = new ObjectOutputStream(System.out)) {
            oos.writeObject(result);
            oos.flush();
            System.err.printf("[GPU-COMPONENT-PROCESS] Output written to stdout%n");
        }
    }
}