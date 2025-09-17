package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Reusable VLM ExecutionPlan Pool to prevent GPU memory exhaustion.
 * 
 * Instead of creating 32 separate VLMTornadoVMMasterPlan instances (one per layer),
 * this pool maintains a single reusable ExecutionPlan that can be updated with
 * different layer weights and executed multiple times.
 * 
 * This solves the GPU memory exhaustion issue where creating thousands of
 * TaskGraphs/ExecutionPlans consumed all 8GB VRAM.
 */
public class VLMExecutionPlanPool {
    
    private final Configuration config;
    private final int visionTokens;
    private final int inputDim;
    private final int kvDim;
    
    // Reusable execution components
    private TornadoExecutionPlan pooledExecutionPlan;
    private GridScheduler pooledGridScheduler;
    private ImmutableTaskGraph pooledTaskGraph;
    
    // Reusable weight arrays (updated per layer)
    private FloatArray pooledKeyWeights;
    private FloatArray pooledValueWeights;
    private FloatArray pooledBatchInput;
    private FloatArray pooledBatchKeyCache;
    private FloatArray pooledBatchValueCache;
    
    private final int weightSize;
    private boolean initialized = false;
    
    public VLMExecutionPlanPool(Configuration config, int visionTokens) {
        this.config = config;
        this.visionTokens = visionTokens;
        this.inputDim = config.dim();
        this.kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        this.weightSize = inputDim * kvDim;
    }
    
    /**
     * Initialize the pooled execution plan (called once)
     */
    public synchronized void initialize(int batchSize) throws Exception {
        if (initialized) return;

        System.err.println("[VLM-POOL] Initializing reusable ExecutionPlan for VLM processing");
        
        // Create pooled weight arrays (will be updated per layer)
        pooledKeyWeights = new FloatArray(weightSize);
        pooledValueWeights = new FloatArray(weightSize);
        pooledBatchInput = new FloatArray(batchSize * inputDim);
        pooledBatchKeyCache = new FloatArray(batchSize * kvDim);
        pooledBatchValueCache = new FloatArray(batchSize * kvDim);
        
        // Create reusable GridScheduler
        pooledGridScheduler = createPooledGridScheduler();
        
        // Create reusable TaskGraph
        pooledTaskGraph = createPooledTaskGraph();
        
        // Create reusable ExecutionPlan
        pooledExecutionPlan = TornadoVMSafeInitializer.createExecutionPlanSafely(pooledTaskGraph);
        pooledExecutionPlan.withGridScheduler(pooledGridScheduler);
        
        initialized = true;
        System.err.println("[VLM-POOL] Reusable ExecutionPlan initialized successfully");
    }
    
    /**
     * Execute VLM processing for a specific layer using the pooled ExecutionPlan
     */
    public synchronized void executeLayer(int layer, 
                                         FloatArray batchInput,
                                         FloatArray keyWeights, 
                                         FloatArray valueWeights,
                                         FloatArray batchKeyCache, 
                                         FloatArray batchValueCache) {
        
        if (!initialized) {
            throw new IllegalStateException("VLMExecutionPlanPool not initialized");
        }
        
        // Update pooled arrays with current layer data
        updatePooledArrays(batchInput, keyWeights, valueWeights, batchKeyCache, batchValueCache);
        
        // Execute using the pooled ExecutionPlan
        pooledExecutionPlan.execute();
        
        // Copy results back to output arrays
        copyResults(batchKeyCache, batchValueCache);
    }
    
    /**
     * Update pooled arrays with current layer weights and data
     */
    private void updatePooledArrays(FloatArray batchInput, 
                                   FloatArray keyWeights, 
                                   FloatArray valueWeights,
                                   FloatArray batchKeyCache, 
                                   FloatArray batchValueCache) {
        
        // Safety checks to prevent bounds overflow
        int maxInputSize = Math.min(batchInput.getSize(), pooledBatchInput.getSize());
        int maxKeyWeightSize = Math.min(keyWeights.getSize(), pooledKeyWeights.getSize());
        int maxValueWeightSize = Math.min(valueWeights.getSize(), pooledValueWeights.getSize());
        int maxKeyCacheSize = Math.min(batchKeyCache.getSize(), pooledBatchKeyCache.getSize());
        int maxValueCacheSize = Math.min(batchValueCache.getSize(), pooledBatchValueCache.getSize());
        
        // Copy input data to pooled arrays with bounds checking
        for (int i = 0; i < maxInputSize; i++) {
            pooledBatchInput.set(i, batchInput.get(i));
        }
        
        for (int i = 0; i < maxKeyWeightSize; i++) {
            pooledKeyWeights.set(i, keyWeights.get(i));
        }
        
        for (int i = 0; i < maxValueWeightSize; i++) {
            pooledValueWeights.set(i, valueWeights.get(i));
        }
        
        // FIXED: Don't clear output arrays - they will be written by the GPU kernel
        // The kernel will overwrite these values, so clearing them prevents the results
        // from being properly computed and copied back
        // for (int i = 0; i < maxKeyCacheSize; i++) {
        //     pooledBatchKeyCache.set(i, 0.0f);
        // }
        
        // for (int i = 0; i < maxValueCacheSize; i++) {
        //     pooledBatchValueCache.set(i, 0.0f);
        // }
        
        // Debug logging for bounds checking
        if (maxInputSize < batchInput.getSize()) {
            System.err.printf("[VLM-POOL] WARNING: Input size truncated from %d to %d%n", 
                             batchInput.getSize(), maxInputSize);
        }
    }
    
    /**
     * Copy results from pooled arrays back to output arrays
     */
    private void copyResults(FloatArray batchKeyCache, FloatArray batchValueCache) {
        // Safety bounds checking for result copying
        int maxKeyCacheSize = Math.min(batchKeyCache.getSize(), pooledBatchKeyCache.getSize());
        int maxValueCacheSize = Math.min(batchValueCache.getSize(), pooledBatchValueCache.getSize());
        
        // Debug: Check if we have non-zero values before copying
        float firstKey = pooledBatchKeyCache.get(0);
        float firstValue = pooledBatchValueCache.get(0);
        if (firstKey != 0.0f || firstValue != 0.0f) {
            System.err.printf("[VLM-POOL-DEBUG] Non-zero results detected: KEY[0]=%f, VALUE[0]=%f%n", 
                            firstKey, firstValue);
        }
        
        // Copy and verify the copy worked
        for (int i = 0; i < maxKeyCacheSize; i++) {
            batchKeyCache.set(i, pooledBatchKeyCache.get(i));
        }
        
        for (int i = 0; i < maxValueCacheSize; i++) {
            batchValueCache.set(i, pooledBatchValueCache.get(i));
        }
        
        // Verify the copy worked by checking the target arrays
        float copiedKey = batchKeyCache.get(0);
        float copiedValue = batchValueCache.get(0);
        System.err.printf("[VLM-POOL-COPY-DEBUG] After copy: batchKeyCache[0]=%f, batchValueCache[0]=%f%n", 
                        copiedKey, copiedValue);
    }
    
    /**
     * Create pooled GridScheduler (reused across all layers)
     */
    private GridScheduler createPooledGridScheduler() {
        GridScheduler scheduler = new GridScheduler();
        
        // FIXED: Use reasonable workgroup count - one workgroup per output dimension
        // Each workgroup processes ALL vision tokens for one output dimension
        WorkerGrid vlmBatchWorker = new WorkerGrid1D(kvDim);
        
        // Match proven working configuration
        vlmBatchWorker.setGlobalWork(kvDim, 1, 1);         // One workgroup per output dimension
        vlmBatchWorker.setLocalWork(8, 1, 1);              // 8 threads per workgroup (GPU limit)
        
        // Map VLM tasks to WorkerGrid (using generic layer naming)
        scheduler.addWorkerGrid("pooled_layer.vlmBatchKeyProjection", vlmBatchWorker);
        scheduler.addWorkerGrid("pooled_layer.vlmBatchValueProjection", vlmBatchWorker);
        
        return scheduler;
    }
    
    /**
     * Create pooled TaskGraph (reused across all layers)
     */
    private ImmutableTaskGraph createPooledTaskGraph() throws Exception {
        // CRITICAL: This method must not be called during static initialization to prevent TornadoVM deadlock
        System.err.println("[VLM-POOL-INIT] Creating pooled TaskGraph - ensuring this is not during static initialization");

        KernelContext context = new KernelContext();

        TaskGraph pooledGraph = TornadoVMSafeInitializer.createTaskGraphSafely("pooled_layer")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, 
                            pooledBatchInput, pooledKeyWeights, pooledValueWeights)
            .task("vlmBatchKeyProjection",
                  TransformerComputeKernelsLayered::vlmBatchKeyProjection,
                  context, pooledBatchInput, pooledKeyWeights, pooledBatchKeyCache,
                  inputDim, kvDim, 8)  // Updated signature: inputDim, kvDim, localWorkGroupSize
            .task("vlmBatchValueProjection",
                  TransformerComputeKernelsLayered::vlmBatchValueProjection,
                  context, pooledBatchInput, pooledValueWeights, pooledBatchValueCache,
                  inputDim, kvDim, 8)  // Updated signature: inputDim, kvDim, localWorkGroupSize
            .transferToHost(DataTransferMode.EVERY_EXECUTION,
                          pooledBatchKeyCache, pooledBatchValueCache);
        
        return pooledGraph.snapshot();
    }
    
    /**
     * Clean up resources (call when done with all VLM processing)
     */
    public synchronized void cleanup() {
        if (!initialized) return;
        
        System.err.println("[VLM-POOL] Cleaning up pooled ExecutionPlan resources");
        
        try {
            if (pooledExecutionPlan != null) {
                // Note: TornadoExecutionPlan doesn't have explicit cleanup in public API
                pooledExecutionPlan = null;
            }
            
            pooledTaskGraph = null;
            pooledGridScheduler = null;
            
            // FloatArrays will be garbage collected
            pooledKeyWeights = null;
            pooledValueWeights = null;
            pooledBatchInput = null;
            pooledBatchKeyCache = null;
            pooledBatchValueCache = null;
            
            initialized = false;
            System.err.println("[VLM-POOL] ExecutionPlan resources cleaned up successfully");
            
        } catch (Exception e) {
            System.err.printf("[VLM-POOL] Warning: Error during cleanup: %s%n", e.getMessage());
        }
    }
}