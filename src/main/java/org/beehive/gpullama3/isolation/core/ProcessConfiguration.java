package org.beehive.gpullama3.isolation.core;

import java.time.Duration;
import java.util.*;

/**
 * Configuration for GPU process isolation including environment, JVM settings, and timeouts.
 */
public class ProcessConfiguration {
    private Duration timeout;
    private SerializationFormat serializationFormat;
    private Map<String, String> environmentVariables;
    private List<String> jvmArguments;
    private int maxRetries;
    private boolean compressionEnabled;
    private String tempDirectory;
    
    public enum SerializationFormat {
        BINARY, JSON, PROTOBUF
    }
    
    private ProcessConfiguration() {
        this.timeout = Duration.ofMinutes(2);
        this.serializationFormat = SerializationFormat.BINARY;
        this.environmentVariables = new HashMap<>();
        this.jvmArguments = new ArrayList<>();
        this.maxRetries = 3;
        this.compressionEnabled = false;
        this.tempDirectory = System.getProperty("java.io.tmpdir");
    }
    
    /**
     * Create default configuration suitable for TornadoVM GPU processes.
     */
    public static ProcessConfiguration defaultConfig() {
        return forTornadoVM();
    }
    
    /**
     * Create TornadoVM-optimized configuration with required environment variables.
     */
    public static ProcessConfiguration forTornadoVM() {
        ProcessConfiguration config = new ProcessConfiguration();
        
        // Copy current environment for TornadoVM
        config.environmentVariables.put("JAVA_HOME", 
            System.getProperty("java.home", System.getenv("JAVA_HOME")));
        config.environmentVariables.put("TORNADO_ROOT", System.getenv("TORNADO_ROOT"));
        config.environmentVariables.put("TORNADO_SDK", System.getenv("TORNADO_SDK"));
        
        // Essential JVM arguments for TornadoVM
        config.jvmArguments.add("--enable-preview");
        config.jvmArguments.add("--add-modules=jdk.incubator.vector");
        config.jvmArguments.add("--add-exports=java.base/jdk.internal.misc=ALL-UNNAMED");
        config.jvmArguments.add("--add-exports=java.base/sun.nio.ch=ALL-UNNAMED");
        
        // Memory settings for GPU operations
        config.jvmArguments.add("-Xmx8g");
        config.jvmArguments.add("-XX:+UseG1GC");
        config.jvmArguments.add("-XX:+UseStringDeduplication");
        
        // TornadoVM-specific settings
        config.jvmArguments.add("-Dtornado.device=0:0");
        config.jvmArguments.add("-Dtornado.print.kernel=false");
        config.jvmArguments.add("-Dtornado.debug=false");
        
        return config;
    }
    
    /**
     * Create configuration for specific component requirements.
     */
    public static ProcessConfiguration forComponent(String componentName) {
        ProcessConfiguration config = forTornadoVM();
        
        // Component-specific optimizations
        switch (componentName.toLowerCase()) {
            case "mlp", "mlpprojector":
                config.timeout = Duration.ofMinutes(3); // MLP can be slow
                config.jvmArguments.add("-Dllava.mlp.process.isolation=true");
                break;
                
            case "vision", "clip":
                config.timeout = Duration.ofSeconds(90); // Vision is faster
                config.jvmArguments.add("-Dllava.vision.process.isolation=true");
                break;
                
            case "moe":
                config.timeout = Duration.ofMinutes(5); // MoE needs more time
                config.maxRetries = 2; // Fewer retries for complex operations
                break;
                
            default:
                // Use default settings
                break;
        }
        
        return config;
    }
    
    // Builder pattern methods for configuration
    public ProcessConfiguration setTimeout(Duration timeout) {
        this.timeout = timeout;
        return this;
    }
    
    public ProcessConfiguration setSerializationFormat(SerializationFormat format) {
        this.serializationFormat = format;
        return this;
    }
    
    public ProcessConfiguration addEnvironmentVariable(String key, String value) {
        if (value != null && !value.isEmpty()) {
            this.environmentVariables.put(key, value);
        }
        return this;
    }
    
    public ProcessConfiguration addJVMArgument(String argument) {
        this.jvmArguments.add(argument);
        return this;
    }
    
    public ProcessConfiguration setMaxRetries(int maxRetries) {
        this.maxRetries = Math.max(1, Math.min(maxRetries, 10)); // Clamp between 1-10
        return this;
    }
    
    public ProcessConfiguration enableCompression() {
        this.compressionEnabled = true;
        return this;
    }
    
    public ProcessConfiguration setTempDirectory(String tempDirectory) {
        this.tempDirectory = tempDirectory;
        return this;
    }
    
    // Read system properties for runtime configuration
    public ProcessConfiguration applySystemProperties() {
        String timeoutProperty = System.getProperty("llama.gpu.isolation.timeout");
        if (timeoutProperty != null) {
            try {
                this.timeout = Duration.parse(timeoutProperty);
            } catch (Exception e) {
                System.err.println("[GPU-ISOLATION] Invalid timeout property: " + timeoutProperty);
            }
        }
        
        String retriesProperty = System.getProperty("llama.gpu.isolation.max.retries");
        if (retriesProperty != null) {
            try {
                this.maxRetries = Integer.parseInt(retriesProperty);
            } catch (NumberFormatException e) {
                System.err.println("[GPU-ISOLATION] Invalid retries property: " + retriesProperty);
            }
        }
        
        String compressionProperty = System.getProperty("llama.gpu.isolation.compression.enabled");
        if ("true".equals(compressionProperty)) {
            this.compressionEnabled = true;
        }
        
        return this;
    }
    
    // Getters
    public Duration getTimeout() { return timeout; }
    public SerializationFormat getSerializationFormat() { return serializationFormat; }
    public Map<String, String> getEnvironmentVariables() { return new HashMap<>(environmentVariables); }
    public List<String> getJvmArguments() { return new ArrayList<>(jvmArguments); }
    public int getMaxRetries() { return maxRetries; }
    public boolean isCompressionEnabled() { return compressionEnabled; }
    public String getTempDirectory() { return tempDirectory; }
    
    @Override
    public String toString() {
        return String.format("ProcessConfiguration{timeout=%s, format=%s, retries=%d, compression=%s}", 
            timeout, serializationFormat, maxRetries, compressionEnabled);
    }
}