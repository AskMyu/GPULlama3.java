package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * Manages batch-aware execution of TornadoVM kernels for SmartCacheArray operations.
 * 
 * This class coordinates kernel execution across multiple FloatArray batches,
 * handling data transfers, batch boundaries, and cross-batch operations.
 * Supports both VLM and non-VLM models with optimal batch scheduling.
 */
public class BatchCacheManager {
    
    private final CacheTopology topology;
    private final FloatArray[] keyBatches;
    private final FloatArray[] valueBatches;
    private final Map<String, TornadoExecutionPlan> executionPlans;
    private final BatchCoordinator coordinator;
    
    // Performance monitoring
    private long totalKernelExecutions = 0;
    private long batchSwitchCount = 0;
    private long crossBatchOperations = 0;
    
    /**
     * Create a batch cache manager for the given topology and cache batches.
     * 
     * @param topology Cache topology analysis
     * @param keyBatches Key cache batches
     * @param valueBatches Value cache batches (can be same as keyBatches for shared cache)
     */
    public BatchCacheManager(CacheTopology topology, FloatArray[] keyBatches, FloatArray[] valueBatches) {
        this.topology = topology;
        this.keyBatches = keyBatches;
        this.valueBatches = valueBatches;
        this.executionPlans = new ConcurrentHashMap<>();
        this.coordinator = new BatchCoordinator(topology);
        
        System.out.printf("[BATCH-MANAGER] Initialized for %s model with %d key batches, %d value batches%n",
                        topology.isVLM ? "VLM" : "LLM", keyBatches.length, valueBatches.length);
        
        // Pre-create execution plans for common operations
        initializeExecutionPlans();
    }
    
    /**
     * Constructor for single cache (key and value share the same batches).
     */
    public BatchCacheManager(CacheTopology topology, FloatArray[] batches) {
        this(topology, batches, batches);
    }
    
    /**
     * Pre-create TornadoVM execution plans for common batch operations.
     */
    private void initializeExecutionPlans() {
        // Create plans for different kernel types
        createCopyToCachePlan();
        createAttentionPlan();
        createCrossBatchPlan();
        
        System.out.printf("[BATCH-MANAGER] Created %d execution plans%n", executionPlans.size());
    }
    
    /**
     * Create execution plan for copyToCache operations.
     */
    private void createCopyToCachePlan() {
        try {
            TaskGraph taskGraph = TornadoVMSafeInitializer.createTaskGraphSafely("copyToCache-batch");
            
            // This is a template - actual data transfers are configured per execution
            taskGraph.task("copyToCacheBatch", BatchCacheManager::copyToCacheBatchKernel, 
                          new FloatArray(1), new FloatArray(1), new FloatArray(1), new FloatArray(1),
                          new IntArray(1), 0, 0, 0);
            
            TornadoExecutionPlan plan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraph.snapshot());
            executionPlans.put("copyToCache", plan);
            
        } catch (Exception e) {
            System.err.printf("[BATCH-MANAGER] Failed to create copyToCache plan: %s%n", e.getMessage());
        }
    }
    
    /**
     * Create execution plan for attention operations.
     */
    private void createAttentionPlan() {
        try {
            TaskGraph taskGraph = TornadoVMSafeInitializer.createTaskGraphSafely("attention-batch");
            
            taskGraph.task("attentionBatch", BatchCacheManager::processAttentionBatchKernel,
                          new FloatArray(1), new FloatArray(1), new FloatArray(1), new FloatArray(1),
                          0, 0, 0, 0, new IntArray(1), 0, 0);
            
            TornadoExecutionPlan plan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraph.snapshot());
            executionPlans.put("attention", plan);
            
        } catch (Exception e) {
            System.err.printf("[BATCH-MANAGER] Failed to create attention plan: %s%n", e.getMessage());
        }
    }
    
    /**
     * Create execution plan for cross-batch operations.
     */
    private void createCrossBatchPlan() {
        try {
            TaskGraph taskGraph = TornadoVMSafeInitializer.createTaskGraphSafely("cross-batch");
            
            taskGraph.task("crossBatchOp", BatchCacheManager::crossBatchKernel,
                          new FloatArray(1), new FloatArray(1), 0, 0, 0);
            
            TornadoExecutionPlan plan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraph.snapshot());
            executionPlans.put("crossBatch", plan);
            
        } catch (Exception e) {
            System.err.printf("[BATCH-MANAGER] Failed to create cross-batch plan: %s%n", e.getMessage());
        }
    }
    
    // ================================================================================
    // MAIN EXECUTION INTERFACE
    // ================================================================================
    
    /**
     * Execute a batched kernel operation with the given arguments.
     * 
     * @param kernelName Name of the kernel to execute
     * @param args Kernel arguments (will be analyzed for batch requirements)
     */
    public void executeBatched(String kernelName, Object... args) {
        totalKernelExecutions++;
        
        switch (kernelName) {
            case "copyToCache":
                executeCopyToCacheBatched(args);
                break;
            case "processHeadsFlashAttention":
                executeAttentionBatched(args);
                break;
            default:
                throw new UnsupportedOperationException("Batched kernel not implemented: " + kernelName);
        }
    }
    
    /**
     * Execute copy-to-cache operation with batch coordination.
     * Args: destKeyCache, srcKey, destValueCache, srcValue, position, kvDim, layer, contextLength
     */
    private void executeCopyToCacheBatched(Object... args) {
        if (args.length < 8) {
            throw new IllegalArgumentException("copyToCache requires 8 arguments");
        }
        
        // Extract arguments
        SmartCacheArray destKeyCache = (SmartCacheArray) args[0];
        FloatArray srcKey = (FloatArray) args[1];
        SmartCacheArray destValueCache = (SmartCacheArray) args[2];
        FloatArray srcValue = (FloatArray) args[3];
        IntArray position = (IntArray) args[4];
        int kvDim = (int) args[5];
        int layer = (int) args[6];
        int contextLength = (int) args[7];
        
        // Calculate cache indices
        int pos = position.get(0);
        long keyIndex = topology.calculateCacheIndex(layer, pos, 0);
        long valueIndex = topology.calculateCacheIndex(layer, pos, 0);
        
        // Determine affected batches
        int keyBatch = destKeyCache.getBatchForIndex((int) keyIndex);
        int valueBatch = destValueCache.getBatchForIndex((int) valueIndex);
        
        if (keyBatch == valueBatch) {
            // Single batch operation (common case)
            executeSingleBatchCopy(keyBatch, destKeyCache, srcKey, destValueCache, srcValue, 
                                 (int) keyIndex, (int) valueIndex, kvDim);
        } else {
            // Cross-batch operation (rare)
            crossBatchOperations++;
            executeCrossBatchCopy(destKeyCache, srcKey, destValueCache, srcValue, 
                                (int) keyIndex, (int) valueIndex, kvDim);
        }
    }
    
    /**
     * Execute attention computation with batch coordination.
     */
    private void executeAttentionBatched(Object... args) {
        // Args: q, keyCache, valueCache, xb, nHeads, headSize, kvDim, kvMul, position, layer, contextLength
        if (args.length < 11) {
            throw new IllegalArgumentException("attention requires 11 arguments");
        }
        
        FloatArray q = (FloatArray) args[0];
        SmartCacheArray keyCache = (SmartCacheArray) args[1];
        SmartCacheArray valueCache = (SmartCacheArray) args[2];
        FloatArray xb = (FloatArray) args[3];
        int nHeads = (int) args[4];
        int headSize = (int) args[5];
        int kvDim = (int) args[6];
        int kvMul = (int) args[7];
        IntArray position = (IntArray) args[8];
        int layer = (int) args[9];
        int contextLength = (int) args[10];
        
        // For attention, we need to access all positions up to current position
        int pos = position.get(0);
        
        if (!keyCache.isBatched()) {
            // Use direct execution for small caches
            executeDirectAttention(q, keyCache, valueCache, xb, nHeads, headSize, 
                                 kvDim, kvMul, position, layer, contextLength);
            return;
        }
        
        // Execute attention across all relevant batches for this layer
        CacheTopology.LayerBatch layerBatch = topology.getBatch(topology.getBatchForLayer(layer));
        
        // Coordinate attention computation across batches
        coordinator.executeAttentionAcrossBatches(layerBatch, q, keyCache, valueCache, xb,
                                                nHeads, headSize, kvDim, kvMul, pos);
    }
    
    // ================================================================================
    // BATCH OPERATION IMPLEMENTATIONS
    // ================================================================================
    
    /**
     * Execute copy operation within a single batch.
     */
    private void executeSingleBatchCopy(int batchIndex, SmartCacheArray destKeyCache, FloatArray srcKey,
                                      SmartCacheArray destValueCache, FloatArray srcValue,
                                      int keyIndex, int valueIndex, int kvDim) {
        try {
            FloatArray keyBatch = destKeyCache.getBatch(batchIndex);
            FloatArray valueBatch = destValueCache.getBatch(batchIndex);
            
            int localKeyIndex = destKeyCache.getLocalIndex(keyIndex);
            int localValueIndex = destValueCache.getLocalIndex(valueIndex);
            
            // Create task graph for this specific operation
            TaskGraph taskGraph = TornadoVMSafeInitializer.createTaskGraphSafely("single-batch-copy")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, srcKey, srcValue)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, keyBatch, valueBatch)
                .task("copyData", BatchCacheManager::copyToCacheBatchKernel,
                      keyBatch, srcKey, valueBatch, srcValue,
                      createIntArray(localKeyIndex), kvDim, 0, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, keyBatch, valueBatch);

            // CRITICAL FIX: Use freeBuffersOnly() instead of try-with-resources to avoid device reset
            TornadoExecutionPlan plan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraph.snapshot());
            try {
                plan.execute();
            } finally {
                // Clean up GPU buffers without device reset
                plan.freeBuffersOnly();
            }
            
        } catch (Exception e) {
            System.err.printf("[BATCH-MANAGER] Single batch copy failed: %s%n", e.getMessage());
            // Fallback to CPU execution
            fallbackCopyOperation(destKeyCache, srcKey, destValueCache, srcValue, keyIndex, valueIndex, kvDim);
        }
    }
    
    /**
     * Execute copy operation across multiple batches.
     */
    private void executeCrossBatchCopy(SmartCacheArray destKeyCache, FloatArray srcKey,
                                     SmartCacheArray destValueCache, FloatArray srcValue,
                                     int keyIndex, int valueIndex, int kvDim) {
        // For cross-batch operations, fall back to element-wise copying
        // This is rare and performance impact is acceptable
        
        for (int i = 0; i < kvDim; i++) {
            destKeyCache.set(keyIndex + i, srcKey.get(i));
            destValueCache.set(valueIndex + i, srcValue.get(i));
        }
    }
    
    /**
     * Execute direct attention for non-batched caches.
     */
    private void executeDirectAttention(FloatArray q, SmartCacheArray keyCache, SmartCacheArray valueCache,
                                      FloatArray xb, int nHeads, int headSize, int kvDim, int kvMul,
                                      IntArray position, int layer, int contextLength) {
        // Direct execution using the cache's single FloatArray
        try {
            TaskGraph taskGraph = TornadoVMSafeInitializer.createTaskGraphSafely("direct-attention")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, q)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, 
                                keyCache.getDirectArray(), valueCache.getDirectArray())
                .task("attention", BatchCacheManager::processAttentionBatchKernel,
                      q, keyCache.getDirectArray(), valueCache.getDirectArray(), xb,
                      nHeads, headSize, kvDim, kvMul, position, layer, contextLength)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);

            // CRITICAL FIX: Use freeBuffersOnly() instead of try-with-resources to avoid device reset
            TornadoExecutionPlan plan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraph.snapshot());
            try {
                plan.execute();
            } finally {
                // Clean up GPU buffers without device reset
                plan.freeBuffersOnly();
            }
            
        } catch (Exception e) {
            System.err.printf("[BATCH-MANAGER] Direct attention failed: %s%n", e.getMessage());
        }
    }
    
    // ================================================================================
    // FALLBACK IMPLEMENTATIONS
    // ================================================================================
    
    /**
     * CPU fallback for copy operations when GPU execution fails.
     */
    private void fallbackCopyOperation(SmartCacheArray destKeyCache, FloatArray srcKey,
                                     SmartCacheArray destValueCache, FloatArray srcValue,
                                     int keyIndex, int valueIndex, int kvDim) {
        // Simple CPU implementation
        for (int i = 0; i < kvDim; i++) {
            destKeyCache.set(keyIndex + i, srcKey.get(i));
            destValueCache.set(valueIndex + i, srcValue.get(i));
        }
    }
    
    // ================================================================================
    // TORNADO VM KERNEL IMPLEMENTATIONS
    // ================================================================================
    
    /**
     * TornadoVM kernel for batch copy operations.
     */
    public static void copyToCacheBatchKernel(FloatArray destKey, FloatArray srcKey,
                                            FloatArray destValue, FloatArray srcValue,
                                            IntArray localIndex, int kvDim, int unused1, int unused2) {
        int idx = localIndex.get(0);
        
        for (int i = 0; i < kvDim; i++) {
            if (idx + i < destKey.getSize() && i < srcKey.getSize()) {
                destKey.set(idx + i, srcKey.get(i));
            }
            if (idx + i < destValue.getSize() && i < srcValue.getSize()) {
                destValue.set(idx + i, srcValue.get(i));
            }
        }
    }
    
    /**
     * TornadoVM kernel for batch attention operations.
     */
    public static void processAttentionBatchKernel(FloatArray q, FloatArray keyCache, FloatArray valueCache,
                                                  FloatArray xb, int nHeads, int headSize, int kvDim,
                                                  int kvMul, IntArray position, int layer, int contextLength) {
        // Simplified attention kernel - full implementation would be more complex
        int pos = position.get(0);
        int layerOffset = layer * contextLength * kvDim;
        
        for (int h = 0; h < nHeads; h++) {
            float score = 0.0f;
            
            // Attention computation (simplified)
            for (int i = 0; i < headSize && i < keyCache.getSize() - layerOffset; i++) {
                if (h * headSize + i < q.getSize()) {
                    score += q.get(h * headSize + i) * keyCache.get(layerOffset + pos * kvDim + i);
                }
            }
            
            // Write to output
            if (h < xb.getSize()) {
                xb.set(h, score);
            }
        }
    }
    
    /**
     * TornadoVM kernel for cross-batch operations.
     */
    public static void crossBatchKernel(FloatArray batch1, FloatArray batch2, 
                                      int offset1, int offset2, int length) {
        for (int i = 0; i < length; i++) {
            if (offset1 + i < batch1.getSize() && offset2 + i < batch2.getSize()) {
                batch2.set(offset2 + i, batch1.get(offset1 + i));
            }
        }
    }
    
    // ================================================================================
    // MONITORING AND DIAGNOSTICS
    // ================================================================================
    
    /**
     * Get performance statistics for this batch manager.
     */
    public BatchManagerStats getStats() {
        return new BatchManagerStats(
            totalKernelExecutions, batchSwitchCount, crossBatchOperations,
            keyBatches.length, valueBatches.length, executionPlans.size()
        );
    }
    
    /**
     * Print comprehensive batch manager diagnostics.
     */
    public void printDiagnostics() {
        BatchManagerStats stats = getStats();
        
        System.out.println("[BATCH-MANAGER] === Diagnostics ===");
        System.out.printf("Total kernel executions: %d%n", stats.totalKernelExecutions);
        System.out.printf("Batch switches: %d%n", stats.batchSwitchCount);
        System.out.printf("Cross-batch operations: %d%n", stats.crossBatchOperations);
        System.out.printf("Key batches: %d, Value batches: %d%n", stats.keyBatches, stats.valueBatches);
        System.out.printf("Execution plans: %d%n", stats.executionPlans);
        
        if (stats.totalKernelExecutions > 0) {
            System.out.printf("Cross-batch ratio: %.2f%%%n", 
                            (stats.crossBatchOperations * 100.0) / stats.totalKernelExecutions);
        }
        
        // Print topology analysis
        topology.printAnalysis();
    }
    
    /**
     * Helper method to create IntArray with proper constructor.
     */
    private static IntArray createIntArray(int value) {
        IntArray arr = new IntArray(1);
        arr.set(0, value);
        return arr;
    }
    
    /**
     * Statistics container for batch manager performance.
     */
    public static class BatchManagerStats {
        public final long totalKernelExecutions;
        public final long batchSwitchCount;
        public final long crossBatchOperations;
        public final int keyBatches;
        public final int valueBatches;
        public final int executionPlans;
        
        public BatchManagerStats(long totalKernelExecutions, long batchSwitchCount, 
                               long crossBatchOperations, int keyBatches, int valueBatches,
                               int executionPlans) {
            this.totalKernelExecutions = totalKernelExecutions;
            this.batchSwitchCount = batchSwitchCount;
            this.crossBatchOperations = crossBatchOperations;
            this.keyBatches = keyBatches;
            this.valueBatches = valueBatches;
            this.executionPlans = executionPlans;
        }
    }
    
    // ================================================================================
    // BATCH COORDINATOR (INNER CLASS)
    // ================================================================================
    
    /**
     * Coordinates complex operations across multiple batches.
     */
    private static class BatchCoordinator {
        private final CacheTopology topology;
        
        public BatchCoordinator(CacheTopology topology) {
            this.topology = topology;
        }
        
        public void executeAttentionAcrossBatches(CacheTopology.LayerBatch layerBatch,
                                                FloatArray q, SmartCacheArray keyCache, SmartCacheArray valueCache,
                                                FloatArray xb, int nHeads, int headSize, int kvDim, int kvMul, int pos) {
            // Simplified implementation - coordinate attention across relevant batches
            // In a full implementation, this would handle complex cross-batch attention patterns
            
            for (int batchIdx = 0; batchIdx < keyCache.getNumBatches(); batchIdx++) {
                CacheTopology.LayerBatch batch = topology.getBatch(batchIdx);
                if (batch.containsLayer(layerBatch.startLayer)) {
                    // Process this batch
                    try {
                        processAttentionBatchKernel(q, keyCache.getBatch(batchIdx), valueCache.getBatch(batchIdx),
                                                  xb, nHeads, headSize, kvDim, kvMul, 
                                                  createIntArray(pos), layerBatch.startLayer, topology.contextLength);
                    } catch (Exception e) {
                        System.err.printf("[BATCH-COORDINATOR] Attention failed for batch %d: %s%n", batchIdx, e.getMessage());
                    }
                }
            }
        }
    }
}