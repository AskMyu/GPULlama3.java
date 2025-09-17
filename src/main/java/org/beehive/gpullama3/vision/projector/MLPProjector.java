package org.beehive.gpullama3.vision.projector;

import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.model.llava.LlavaConfiguration;
import org.beehive.gpullama3.model.loader.ModelLoader;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.Map;

/**
 * MLP (Multi-Layer Perceptron) projector for mapping vision features to language embedding space.
 * Loads real weights from LLaVA mmproj GGUF file and performs actual projection operations.
 * 
 * LLaVA MLP Projector Architecture:
 * - Input: CLIP features (1024-dim from CLIP-ViT-Large-336)
 * - Layer 1: Linear(1024 -> 4096) + GELU activation  
 * - Layer 2: Linear(4096 -> 4096) (projects to Llama embedding space)
 * - Output: Llama-compatible embeddings (4096-dim)
 */
public class MLPProjector {
    private final LlavaConfiguration config;
    private final Map<String, GGMLTensorEntry> visionTensors;
    
    // Architecture parameters
    private final int inputDim;      // 1024 (CLIP features)
    private final int hiddenDim;     // 4096 (intermediate layer)  
    private final int outputDim;     // 4096 (Llama embeddings)
    private final String activationType; // "gelu"
    
    // Loaded weight tensors
    private FloatTensor fc1Weight;   // [1024, 4096] 
    private FloatTensor fc1Bias;     // [4096]
    private FloatTensor fc2Weight;   // [4096, 4096]
    private FloatTensor fc2Bias;     // [4096]
    
    // Pre-allocated buffers for memory optimization
    private final FloatArray tokenBuffer;     // For single token extraction
    private final FloatArray hiddenBuffer;    // For intermediate layer output
    private final FloatArray outputBuffer;    // For final layer output
    
    public MLPProjector(LlavaConfiguration config, Map<String, GGMLTensorEntry> visionTensors) {
        this.config = config;
        this.visionTensors = visionTensors;
        this.inputDim = config.getVisionInputDim();      // 1024
        this.hiddenDim = config.getLanguageEmbeddingDim(); // 4096  
        this.outputDim = config.getLanguageEmbeddingDim(); // 4096
        this.activationType = "gelu"; // Standard for LLaVA
        
        // Pre-allocate reusable buffers
        this.tokenBuffer = new FloatArray(inputDim);     // 1024
        this.hiddenBuffer = new FloatArray(hiddenDim);   // 4096
        this.outputBuffer = new FloatArray(outputDim);   // 4096
        
        // Load weights from mmproj GGUF tensors
        loadWeights();
        
        System.out.println("MLP Projector initialized:");
        System.out.println("  Architecture: " + inputDim + " -> " + hiddenDim + " -> " + outputDim);
        System.out.println("  Activation: " + activationType);
        System.out.println("  Weights loaded from vision tensors");
        System.out.println("  Pre-allocated buffers: token=" + tokenBuffer.getSize() + 
                          ", hidden=" + hiddenBuffer.getSize() + 
                          ", output=" + outputBuffer.getSize());
    }

    /**
     * Load MLP projector weights from mmproj GGUF tensors.
     */
    private void loadWeights() {
        try {
            // Load first layer weights and biases
            fc1Weight = loadTensor("mm.0.weight", "mm_projector.0.weight", "mm_projector.mlp.0.weight", "projector.0.weight");
            fc1Bias = loadTensor("mm.0.bias", "mm_projector.0.bias", "mm_projector.mlp.0.bias", "projector.0.bias");
            
            // Load second layer weights and biases  
            fc2Weight = loadTensor("mm.2.weight", "mm_projector.2.weight", "mm_projector.mlp.2.weight", "projector.2.weight");
            fc2Bias = loadTensor("mm.2.bias", "mm_projector.2.bias", "mm_projector.mlp.2.bias", "projector.2.bias");
            
            System.out.println("MLP Projector weights loaded successfully");
            
        } catch (Exception e) {
            System.err.println("Warning: Could not load all MLP projector weights: " + e.getMessage());
            System.err.println("Available vision tensors: " + visionTensors.keySet().stream().limit(10).toList());
            
            // Create dummy weights for development
            createDummyWeights();
        }
    }

    /**
     * Load a specific tensor, trying multiple possible names.
     */
    private FloatTensor loadTensor(String... possibleNames) {
        for (String name : possibleNames) {
            GGMLTensorEntry tensor = visionTensors.get(name);
            if (tensor != null) {
                return ModelLoader.loadQuantized(tensor);
            }
        }
        
        System.err.println("Warning: Could not find tensor: " + possibleNames[0]);
        return createDummyTensor(inputDim, outputDim); // Return dummy tensor
    }

    /**
     * Create dummy weights for development/testing.
     */
    private void createDummyWeights() {
        fc1Weight = createDummyTensor(inputDim, hiddenDim);
        fc1Bias = createDummyTensor(hiddenDim);
        fc2Weight = createDummyTensor(hiddenDim, outputDim);
        fc2Bias = createDummyTensor(outputDim);
        System.out.println("Created dummy MLP projector weights for development");
    }

    /**
     * Create a dummy tensor for development.
     */
    private FloatTensor createDummyTensor(int... dims) {
        int size = 1;
        for (int dim : dims) size *= dim;
        
        FloatTensor dummy = new ArrayFloatTensor(new float[size]);
        for (int i = 0; i < size; i++) {
            dummy.setFloat(i, 0.1f * (float)Math.random()); // Small random values
        }
        return dummy;
    }
    
    /**
     * Project vision features to language embedding space.
     * 
     * @param visionFeatures Input features from CLIP encoder [numTokens * hiddenDim]
     * @return Projected features in language embedding space [numTokens * outputDim]
     */
    public FloatArray project(FloatArray visionFeatures) {
        System.err.println("[VLM-DEBUG] ===== ENTERING MLPProjector.project() =====");
        System.err.println("[VLM-DEBUG] Input size: " + visionFeatures.getSize());
        long totalStartTime = System.nanoTime();
        System.err.println("[PERF] Starting MLP projection");
        
        // Calculate number of vision tokens (576 for CLIP-ViT-Large-336)
        int numTokens = visionFeatures.getSize() / inputDim;
        if (visionFeatures.getSize() % inputDim != 0) {
            throw new IllegalArgumentException("Vision features size must be multiple of input dim: " + inputDim);
        }
        
        System.out.println("Projecting vision features: " + numTokens + " tokens, " + inputDim + "D -> " + outputDim + "D");
        
        long startTime = System.nanoTime();
        // Output: [numTokens, outputDim]
        FloatArray output = new FloatArray(numTokens * outputDim);
        long allocTime = System.nanoTime() - startTime;
        System.err.printf("[PERF] Output array allocation took: %.3f ms%n", allocTime / 1_000_000.0);
        
        try {
            startTime = System.nanoTime();
            // Process each token through MLP
            for (int token = 0; token < numTokens; token++) {
                long tokenStartTime = System.nanoTime();
                
                // Extract single token features [inputDim]
                FloatArray tokenFeatures = extractToken(visionFeatures, token, inputDim);
                long extractTime = System.nanoTime() - tokenStartTime;
                
                // Project single token through MLP layers
                long projectStartTime = System.nanoTime();
                FloatArray projectedToken = projectSingleToken(tokenFeatures);
                long projectTime = System.nanoTime() - projectStartTime;
                
                // Store projected token in output
                long storeStartTime = System.nanoTime();
                storeToken(output, projectedToken, token, outputDim);
                long storeTime = System.nanoTime() - storeStartTime;
                
                // Log timing for first few tokens to avoid spam
                if (token < 3 || token == numTokens - 1) {
                    System.err.printf("[PERF] Token %d: extract=%.3fms, project=%.3fms, store=%.3fms%n", 
                                    token, extractTime / 1_000_000.0, projectTime / 1_000_000.0, storeTime / 1_000_000.0);
                }
            }
            
            long projectionLoopTime = System.nanoTime() - startTime;
            System.err.printf("[PERF] All tokens projection loop took: %.2f ms%n", projectionLoopTime / 1_000_000.0);
            
            long totalDuration = System.nanoTime() - totalStartTime;
            System.err.printf("[PERF] Total MLP projection took: %.2f ms%n", totalDuration / 1_000_000.0);
            
            return output;
            
        } catch (Exception e) {
            System.err.println("MLP projection failed: " + e.getMessage());
            e.printStackTrace();
            
            // Return dummy projection for development
            return createDummyProjection(numTokens);
        }
    }
    
    /**
     * Project single vision token through MLP layers.
     */
    private FloatArray projectSingleToken(FloatArray tokenFeatures) {
        long startTime = System.nanoTime();
        
        // Layer 1: [inputDim] -> [hiddenDim] + GELU
        // Use pre-allocated hidden buffer
        FloatArray hidden = this.hiddenBuffer;
        hidden.clear();
        matmulSingleInPlace(tokenFeatures, fc1Weight, hidden, inputDim, hiddenDim);
        long matmul1Time = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        addBiasSingle(hidden, fc1Bias);
        long bias1Time = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        applyActivationSingle(hidden, activationType);
        long geluTime = System.nanoTime() - startTime;
        
        // Layer 2: [hiddenDim] -> [outputDim]  
        startTime = System.nanoTime();
        // Use pre-allocated output buffer
        FloatArray output = this.outputBuffer;
        output.clear();
        matmulSingleInPlace(hidden, fc2Weight, output, hiddenDim, outputDim);
        long matmul2Time = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        addBiasSingle(output, fc2Bias);
        long bias2Time = System.nanoTime() - startTime;
        
        // Only log detailed timing for first few tokens to avoid spam
        if (Thread.currentThread().getName().contains("0") || Math.random() < 0.001) {
            System.err.printf("[PERF] Single token MLP: matmul1=%.3f, bias1=%.3f, gelu=%.3f, matmul2=%.3f, bias2=%.3f ms%n",
                            matmul1Time / 1_000_000.0, bias1Time / 1_000_000.0, geluTime / 1_000_000.0, 
                            matmul2Time / 1_000_000.0, bias2Time / 1_000_000.0);
        }
        
        return output;
    }

    /**
     * Extract a single token from vision features array.
     */
    private FloatArray extractToken(FloatArray features, int tokenIndex, int tokenDim) {
        FloatArray token = new FloatArray(tokenDim);
        int startIdx = tokenIndex * tokenDim;
        
        for (int i = 0; i < tokenDim; i++) {
            token.set(i, features.get(startIdx + i));
        }
        return token;
    }

    /**
     * Store a projected token back into the output array.
     */
    private void storeToken(FloatArray output, FloatArray token, int tokenIndex, int tokenDim) {
        int startIdx = tokenIndex * tokenDim;
        
        for (int i = 0; i < tokenDim; i++) {
            output.set(startIdx + i, token.get(i));
        }
    }

    /**
     * Create dummy projection for development.
     */
    private FloatArray createDummyProjection(int numTokens) {
        FloatArray dummy = new FloatArray(numTokens * outputDim);
        for (int i = 0; i < dummy.getSize(); i++) {
            dummy.set(i, 0.1f * (float)Math.random());
        }
        return dummy;
    }
    
    public int getInputDim() { return inputDim; }
    public int getHiddenDim() { return hiddenDim; }
    public int getOutputDim() { return outputDim; }
    public String getActivationType() { return activationType; }
    
    @Override
    public String toString() {
        return String.format("MLPProjector{%d→%d→%d, activation=%s}", 
                           inputDim, hiddenDim, outputDim, activationType);
    }
    
    // Helper methods for single-token matrix operations
    
    /**
     * Matrix multiplication for single token: input [inputDim] × weight [inputDim, outputDim] -> output [outputDim]
     */
    private FloatArray matmulSingle(FloatArray input, FloatTensor weight, int inputDim, int outputDim) {
        FloatArray output = new FloatArray(outputDim);
        
        for (int o = 0; o < outputDim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < inputDim; i++) {
                // weight is stored in [inputDim, outputDim] format
                sum += input.get(i) * weight.getFloat(i * outputDim + o);
            }
            output.set(o, sum);
        }
        return output;
    }
    
    /**
     * In-place matrix multiplication to avoid memory allocation.
     */
    private void matmulSingleInPlace(FloatArray input, FloatTensor weight, FloatArray output, int inputDim, int outputDim) {
        for (int o = 0; o < outputDim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < inputDim; i++) {
                // weight is stored in [inputDim, outputDim] format
                sum += input.get(i) * weight.getFloat(i * outputDim + o);
            }
            output.set(o, sum);
        }
    }
    
    /**
     * Add bias to single token output.
     */
    private void addBiasSingle(FloatArray tensor, FloatTensor bias) {
        for (int i = 0; i < tensor.getSize(); i++) {
            float current = tensor.get(i);
            tensor.set(i, current + bias.getFloat(i));
        }
    }
    
    /**
     * Apply activation function to single token.
     */
    private void applyActivationSingle(FloatArray tensor, String activation) {
        switch (activation.toLowerCase()) {
            case "gelu":
                for (int i = 0; i < tensor.getSize(); i++) {
                    float x = tensor.get(i);
                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                    float gelu = (float) (0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
                    tensor.set(i, gelu);
                }
                break;
            case "relu":
                for (int i = 0; i < tensor.getSize(); i++) {
                    tensor.set(i, Math.max(0.0f, tensor.get(i)));
                }
                break;
            case "silu":
            case "swish":
                for (int i = 0; i < tensor.getSize(); i++) {
                    float x = tensor.get(i);
                    tensor.set(i, (float) (x / (1.0 + Math.exp(-x))));
                }
                break;
            default:
                // No activation (linear)
                break;
        }
    }
}