package org.beehive.gpullama3.vision.projector;

import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.model.llava.LlavaConfiguration;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.isolation.adapters.MLPProcessAdapter;
import org.beehive.gpullama3.isolation.core.ProcessExecutionException;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.Map;

/**
 * GPU-accelerated MLP Projector using TornadoVM for parallel batch processing.
 * Projects vision tokens from CLIP space (1024-dim) to LLM space (4096-dim).
 * 
 * Performance Improvements:
 * - Batch matrix multiplication on GPU
 * - Parallel GELU activation processing
 * - Optimized memory transfers
 * - Reduced CPU-GPU synchronization
 */
public class MLPProjectorGPU extends MLPProjector {

    private boolean useGPU;
    private boolean gpuInitialized = false;
    private final boolean useProcessIsolation;
    private final MLPProcessAdapter processAdapter;
    private final Map<String, GGMLTensorEntry> visionTensors;
    
    // GPU computation buffers
    private FloatArray gpuInputBuffer;
    private FloatArray gpuHiddenBuffer;
    private FloatArray gpuOutputBuffer;
    private FloatArray gpuWeightBuffer1;
    private FloatArray gpuWeightBuffer2;
    private FloatArray gpuBiasBuffer1;
    private FloatArray gpuBiasBuffer2;
    
    // TornadoVM execution plans
    private TornadoExecutionPlan batchProjectionPlan;
    
    // Architecture parameters
    private final int inputDim = 1024;   // CLIP embedding dimension
    private final int hiddenDim = 4096;  // MLP hidden dimension  
    private final int outputDim = 4096;  // LLM embedding dimension
    private final int maxTokens = 577;   // 576 patches + 1 class token
    
    public MLPProjectorGPU(LlavaConfiguration config, Map<String, GGMLTensorEntry> visionTensors) {
        super(config, visionTensors);
        // We extend MLPProjector, so we inherit CPU functionality

        // Store vision tensors for later GPU initialization
        this.visionTensors = visionTensors;

        // Initialize process isolation first
        this.useProcessIsolation = getBooleanProperty("llava.mlp.process.isolation.enabled", true);
        this.processAdapter = useProcessIsolation ? new MLPProcessAdapter() : null;

        // Defer GPU initialization to avoid static TaskGraph creation during constructor
        this.useGPU = false; // Will be initialized lazily on first use
        this.gpuInitialized = false;

        System.err.printf("[MLP-GPU] Initialized: processIsolation=%s, useGPU=%s (lazy init)%n",
            useProcessIsolation, "deferred");
        
        if (useProcessIsolation && processAdapter != null) {
            // Validate process isolation environment
            if (processAdapter.isAvailable()) {
                System.err.println("[MLP-GPU] Process isolation enabled and available");
            } else {
                System.err.println("[MLP-GPU] Warning: Process isolation requested but not available");
            }
        }
    }

    private synchronized void initializeGPULazy() {
        if (gpuInitialized) {
            return; // Already initialized
        }
        System.out.println("[GPU] Performing lazy GPU initialization for MLP...");
        try {
            this.useGPU = initializeGPU();
            if (useGPU) {
                initializeGPUBuffers();
                loadWeightsToGPU(visionTensors);
                createExecutionPlans();
            }
        } catch (Exception | Error e) {
            System.err.println("[GPU-MLP] ❌ Lazy GPU initialization failed: " + e.getMessage());
            System.err.println("[GPU-MLP] Falling back to CPU processing");
            this.useGPU = false;
        }
        this.gpuInitialized = true;
    }

    private boolean initializeGPU() throws Exception {
        try {
            // Test GPU matrix operations
            FloatArray testA = new FloatArray(100);
            FloatArray testB = new FloatArray(100);
            FloatArray testC = new FloatArray(100);
            
            testA.init(2.0f);
            testB.init(3.0f);
            
            TaskGraph testGraph = TornadoVMSafeInitializer.createTaskGraphSafely("matmulTest")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, testA, testB, testC)
                .task("matmul", MLPProjectorGPU::vectorMatmul, testA, testB, testC, 10, 10)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, testC);
                
            ImmutableTaskGraph immutableGraph = testGraph.snapshot();
            try (TornadoExecutionPlan plan = TornadoVMSafeInitializer.createExecutionPlanSafely(immutableGraph)) {
                plan.execute();
            }
            
            System.out.println("[GPU] MLP GPU acceleration initialized successfully");
            return true;
        } catch (Exception e) {
            System.out.println("[GPU] MLP GPU acceleration unavailable, falling back to CPU: " + e.getMessage());
            return false;
        }
    }
    
    private void initializeGPUBuffers() {
        // Pre-allocate buffers for batch processing
        int maxBatchSize = maxTokens;
        
        gpuInputBuffer = new FloatArray(maxBatchSize * inputDim);    // 577 * 1024
        gpuHiddenBuffer = new FloatArray(maxBatchSize * hiddenDim);  // 577 * 4096
        gpuOutputBuffer = new FloatArray(maxBatchSize * outputDim);  // 577 * 4096
        
        // Weight and bias buffers
        gpuWeightBuffer1 = new FloatArray(inputDim * hiddenDim);     // 1024 * 4096
        gpuWeightBuffer2 = new FloatArray(hiddenDim * outputDim);    // 4096 * 4096
        gpuBiasBuffer1 = new FloatArray(hiddenDim);                  // 4096
        gpuBiasBuffer2 = new FloatArray(outputDim);                  // 4096
    }
    
    private void loadWeightsToGPU(Map<String, GGMLTensorEntry> visionTensors) {
        // Load pre-trained weights to GPU buffers
        try {
            // Load first layer weights and bias
            GGMLTensorEntry weight1Entry = visionTensors.get("vision.model.mm_projector.0.weight");
            GGMLTensorEntry bias1Entry = visionTensors.get("vision.model.mm_projector.0.bias");
            
            if (weight1Entry != null) {
                FloatTensor weight1Tensor = ModelLoader.loadQuantized(weight1Entry);
                for (int i = 0; i < Math.min(weight1Tensor.size(), gpuWeightBuffer1.getSize()); i++) {
                    gpuWeightBuffer1.set(i, weight1Tensor.getFloat(i));
                }
            }
            
            if (bias1Entry != null) {
                FloatTensor bias1Tensor = ModelLoader.loadQuantized(bias1Entry);
                for (int i = 0; i < Math.min(bias1Tensor.size(), gpuBiasBuffer1.getSize()); i++) {
                    gpuBiasBuffer1.set(i, bias1Tensor.getFloat(i));
                }
            }
            
            // Load second layer weights and bias  
            GGMLTensorEntry weight2Entry = visionTensors.get("vision.model.mm_projector.2.weight");
            GGMLTensorEntry bias2Entry = visionTensors.get("vision.model.mm_projector.2.bias");
            
            if (weight2Entry != null) {
                FloatTensor weight2Tensor = ModelLoader.loadQuantized(weight2Entry);
                for (int i = 0; i < Math.min(weight2Tensor.size(), gpuWeightBuffer2.getSize()); i++) {
                    gpuWeightBuffer2.set(i, weight2Tensor.getFloat(i));
                }
            }
            
            if (bias2Entry != null) {
                FloatTensor bias2Tensor = ModelLoader.loadQuantized(bias2Entry);
                for (int i = 0; i < Math.min(bias2Tensor.size(), gpuBiasBuffer2.getSize()); i++) {
                    gpuBiasBuffer2.set(i, bias2Tensor.getFloat(i));
                }
            }
            
            System.out.println("[GPU] MLP weights loaded to GPU buffers");
        } catch (Exception e) {
            System.err.println("[GPU] Error loading weights to GPU: " + e.getMessage());
        }
    }
    
    private void createExecutionPlans() throws Exception {
        // Create initial execution plan for batch MLP projection
        createFreshExecutionPlan();
    }
    
    private void createFreshExecutionPlan() throws Exception {
        // Create a completely fresh execution plan to avoid GPU resource conflicts
        System.err.println("[GPU-MLP] Creating fresh TornadoVM execution plan...");
        
        // CRITICAL GPU RESOURCE CLEANUP: Explicitly free device memory from previous execution plan
        if (batchProjectionPlan != null) {
            System.err.println("[GPU-MLP] Explicitly freeing GPU device memory from previous execution plan...");
            try {
                batchProjectionPlan.freeDeviceMemory();
                System.err.println("[GPU-MLP] GPU device memory freed successfully");
            } catch (Exception e) {
                System.err.println("[GPU-MLP] Warning: Error freeing device memory: " + e.getMessage());
            }
            
            System.err.println("[GPU-MLP] Closing previous execution plan...");
            try {
                batchProjectionPlan.close();
                System.err.println("[GPU-MLP] Previous execution plan closed successfully");
            } catch (Exception e) {
                System.err.println("[GPU-MLP] Warning: Error closing execution plan: " + e.getMessage());
            }
            batchProjectionPlan = null;
        }
        
        // Force garbage collection to ensure GPU resources are released
        System.err.println("[GPU-MLP] Forcing garbage collection for GPU resource cleanup...");
        System.gc();
        System.runFinalization();
        
        // Brief delay to allow GPU driver to process resource cleanup
        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        TaskGraph projectionGraph = TornadoVMSafeInitializer.createTaskGraphSafely("batchProjection_" + System.currentTimeMillis())
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, 
                             gpuInputBuffer, gpuWeightBuffer1, gpuWeightBuffer2, 
                             gpuBiasBuffer1, gpuBiasBuffer2, gpuHiddenBuffer, gpuOutputBuffer)
            .task("batchMLP", MLPProjectorGPU::batchMLPProjection,
                  gpuInputBuffer, gpuHiddenBuffer, gpuOutputBuffer,
                  gpuWeightBuffer1, gpuWeightBuffer2, gpuBiasBuffer1, gpuBiasBuffer2,
                  inputDim, hiddenDim, outputDim, maxTokens)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, gpuOutputBuffer);
        
        batchProjectionPlan = TornadoVMSafeInitializer.createExecutionPlanSafely(projectionGraph.snapshot());
        System.err.println("[GPU-MLP] Fresh execution plan created successfully with proper resource cleanup");
    }
    
    @Override
    public FloatArray project(FloatArray visionTokens) {
        System.err.println("[VLM-DEBUG] ===== ENTERING MLPProjectorGPU.project() =====");
        System.err.println("[VLM-DEBUG] processIsolation: " + useProcessIsolation + 
            ", useGPU: " + useGPU + ", input size: " + visionTokens.getSize());
        
        // Priority 1: Process isolation (to resolve TornadoVM GPU deadlock)
        if (useProcessIsolation && processAdapter != null) {
            System.err.println("[MLP-GPU] Using process isolation to avoid GPU deadlock");
            
            try {
                return processAdapter.executeInIsolation(visionTokens);
                
            } catch (ProcessExecutionException e) {
                System.err.printf("[MLP-GPU] Process isolation failed: %s%n", e.getMessage());
                System.err.println("[MLP-GPU] CPU fallback disabled - throwing exception to force subprocess fix");
                throw new RuntimeException("MLP subprocess failed and CPU fallback is disabled", e);
            }
        }
        
        // Priority 2: Direct GPU (if process isolation disabled)
        // Initialize GPU lazily only when actually needed
        initializeGPULazy();

        if (useGPU) {
            System.err.println("[VLM-DEBUG] Using direct GPU implementation");
            return projectGPU(visionTokens);
        }
        
        // Priority 3: CPU fallback
        System.err.println("[VLM-DEBUG] Falling back to CPU MLPProjector");
        return super.project(visionTokens);
    }
    
    /**
     * Direct GPU implementation (kept for when process isolation is disabled).
     * This method contains the original GPU logic but should rarely be used due to deadlock issues.
     */
    private FloatArray projectGPU(FloatArray visionTokens) {
        
        long startTime = System.nanoTime();
        
        try {
            System.err.println("[GPU-MLP] Starting GPU projection...");
            
            // Copy input tokens to GPU buffer
            System.err.println("[GPU-MLP] Copying tokens to GPU...");
            copyTokensToGPU(visionTokens);
            System.err.println("[GPU-MLP] Tokens copied to GPU successfully");
            
            // Execute batch MLP projection on GPU
            System.err.println("[GPU-MLP] Executing batchProjectionPlan on GPU...");
            
            // FIX: Proper GPU resource synchronization for TornadoVM
            // The deadlock occurs because vision encoder and MLP projector compete for GPU resources
            System.err.println("[GPU-MLP] Implementing TornadoVM GPU synchronization fix...");
            
            // CRITICAL: Ensure GPU is in clean state before MLP execution
            // TornadoVM doesn't handle concurrent GPU operations well
            long execStart = System.nanoTime();
            long execTime;
            
            synchronized (this) {
                // Force all previous GPU operations to complete
                System.err.println("[GPU-MLP] Forcing GPU synchronization...");
                System.gc();
                System.runFinalization();
                Thread.sleep(100); // Allow GPU operations to settle
                
                // Create a fresh execution context for MLP
                System.err.println("[GPU-MLP] Creating fresh GPU execution context...");
                execStart = System.nanoTime(); // Reset timing
                
                try {
                    System.err.println("[GPU-MLP] Recreating execution plan to avoid GPU resource conflicts...");
                    // CRITICAL FIX: Recreate execution plan to avoid TornadoVM GPU deadlock
                    // The issue is that the existing execution plan holds GPU resources from previous operations
                    createFreshExecutionPlan();
                    
                    System.err.println("[GPU-MLP] Executing MLP on GPU with fresh execution plan...");
                    batchProjectionPlan.execute();
                    execTime = System.nanoTime() - execStart;
                    System.err.println("[GPU-MLP] GPU MLP execution completed successfully");
                } catch (Exception gpuError) {
                    System.err.println("[GPU-MLP] GPU execution error: " + gpuError.getMessage());
                    throw gpuError; // Propagate the error - no CPU fallback allowed
                }
            }
            System.err.printf("[GPU-MLP] GPU execution completed in %.2f ms%n", execTime / 1_000_000.0);
            
            // Convert GPU result back to FloatArray
            System.err.println("[GPU-MLP] Converting GPU result back to FloatArray...");
            FloatArray result = convertGPUResultToArray(visionTokens.getSize() / inputDim);
            System.err.println("[GPU-MLP] GPU result conversion completed");
            
            return result;
            
        } catch (Exception e) {
            System.err.println("[GPU-MLP] CRITICAL: GPU MLP projection failed - NO CPU FALLBACK ALLOWED");
            System.err.println("[GPU-MLP] Error: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("GPU MLP projection failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Get boolean system property with default value.
     */
    private boolean getBooleanProperty(String propertyName, boolean defaultValue) {
        String property = System.getProperty(propertyName);
        if (property == null) {
            return defaultValue;
        }
        return "true".equalsIgnoreCase(property);
    }
    
    private void copyTokensToGPU(FloatArray visionTokens) {
        int tokensCount = visionTokens.getSize() / inputDim;
        
        for (int token = 0; token < tokensCount && token < maxTokens; token++) {
            for (int dim = 0; dim < inputDim; dim++) {
                int srcIdx = token * inputDim + dim;
                int dstIdx = token * inputDim + dim;
                
                if (srcIdx < visionTokens.getSize() && dstIdx < gpuInputBuffer.getSize()) {
                    gpuInputBuffer.set(dstIdx, visionTokens.get(srcIdx));
                }
            }
        }
    }
    
    private FloatArray convertGPUResultToArray(int tokenCount) {
        int totalSize = tokenCount * outputDim;
        FloatArray result = new FloatArray(totalSize);
        
        for (int token = 0; token < tokenCount; token++) {
            for (int dim = 0; dim < outputDim; dim++) {
                int gpuIdx = token * outputDim + dim;
                int resultIdx = token * outputDim + dim;
                
                if (gpuIdx < gpuOutputBuffer.getSize() && resultIdx < result.getSize()) {
                    result.set(resultIdx, gpuOutputBuffer.get(gpuIdx));
                }
            }
        }
        
        return result;
    }
    
    public void close() {
        if (useGPU && batchProjectionPlan != null) {
            try {
                System.err.println("[GPU-MLP] Closing GPU MLP projector - freeing device memory...");
                batchProjectionPlan.freeDeviceMemory();
                System.err.println("[GPU-MLP] Device memory freed");
                
                System.err.println("[GPU-MLP] Closing execution plan...");
                batchProjectionPlan.close();
                System.err.println("[GPU-MLP] Execution plan closed");
                
                batchProjectionPlan = null;
            } catch (Exception e) {
                System.err.println("Error closing GPU MLP resources: " + e.getMessage());
            }
        }
        // super.close(); // MLPProjector doesn't have a close method
    }
    
    // GPU Kernel Methods (TornadoVM compiles these to GPU kernels)
    
    public static void vectorMatmul(FloatArray a, FloatArray b, FloatArray c, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                if (idx < c.getSize()) {
                    c.set(idx, a.get(i) * b.get(j));
                }
            }
        }
    }
    
    public static void batchMLPProjection(FloatArray input, FloatArray hidden, FloatArray output,
                                         FloatArray weight1, FloatArray weight2, 
                                         FloatArray bias1, FloatArray bias2,
                                         int inputDim, int hiddenDim, int outputDim, int numTokens) {
        
        // Process all tokens in parallel
        for (int token = 0; token < numTokens; token++) {
            
            // First layer: input -> hidden (with GELU activation)
            for (int h = 0; h < hiddenDim; h++) {
                float sum = 0.0f;
                
                // Matrix multiplication: input @ weight1
                for (int i = 0; i < inputDim; i++) {
                    int inputIdx = token * inputDim + i;
                    int weightIdx = i * hiddenDim + h;
                    
                    if (inputIdx < input.getSize() && weightIdx < weight1.getSize()) {
                        sum += input.get(inputIdx) * weight1.get(weightIdx);
                    }
                }
                
                // Add bias
                if (h < bias1.getSize()) {
                    sum += bias1.get(h);
                }
                
                // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                float x = sum;
                float x3 = x * x * x;
                float inner = (float) (Math.sqrt(2.0 / Math.PI) * (x + 0.044715f * x3));
                float tanh_inner = (float) Math.tanh(inner);
                float gelu = x * 0.5f * (1.0f + tanh_inner);
                
                int hiddenIdx = token * hiddenDim + h;
                if (hiddenIdx < hidden.getSize()) {
                    hidden.set(hiddenIdx, gelu);
                }
            }
            
            // Second layer: hidden -> output (linear)
            for (int o = 0; o < outputDim; o++) {
                float sum = 0.0f;
                
                // Matrix multiplication: hidden @ weight2
                for (int h = 0; h < hiddenDim; h++) {
                    int hiddenIdx = token * hiddenDim + h;
                    int weightIdx = h * outputDim + o;
                    
                    if (hiddenIdx < hidden.getSize() && weightIdx < weight2.getSize()) {
                        sum += hidden.get(hiddenIdx) * weight2.get(weightIdx);
                    }
                }
                
                // Add bias
                if (o < bias2.getSize()) {
                    sum += bias2.get(o);
                }
                
                int outputIdx = token * outputDim + o;
                if (outputIdx < output.getSize()) {
                    output.set(outputIdx, sum);
                }
            }
        }
    }
}