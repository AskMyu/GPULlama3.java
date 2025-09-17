package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.ArrayList;
import java.util.List;

/**
 * VLM-specific TornadoVM Layer Planner
 * 
 * Creates properly synchronized TaskGraphs and GridScheduler for VLM batch processing
 * following the proven TornadoVMMasterPlan architectural pattern.
 * 
 * This planner generates VLM-specific TaskGraphs that integrate seamlessly with
 * the TornadoVM execution framework, solving the TaskGraph/GridScheduler synchronization
 * issues that occurred with independent TaskGraph creation.
 */
public class VLMTornadoVMLayerPlanner {
    
    private final Configuration config;
    private final int visionTokens;
    private final int inputDim;
    private final int kvDim;
    
    public VLMTornadoVMLayerPlanner(Configuration config, int visionTokens) {
        this.config = config;
        this.visionTokens = visionTokens; // 144 for vision
        this.inputDim = config.dim();
        this.kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
    }
    
    /**
     * Creates VLM TaskGraph and GridScheduler for a specific layer using the proven TornadoVMMasterPlan pattern
     * 
     * This method follows the same architectural pattern as working non-VLM implementations:
     * 1. Create TaskGraph for the specific layer's VLM processing
     * 2. Configure GridScheduler with proper WorkerGrid mapping
     * 3. Return synchronized Tuple2<TaskGraph, GridScheduler>
     * 
     * @param layer Layer index to process
     * @param batchInput Vision embeddings [batchSize, inputDim]
     * @param keyWeights Key projection weights [kvDim, inputDim]
     * @param valueWeights Value projection weights [kvDim, inputDim]
     * @param batchKeyCache Key cache output [batchSize, kvDim]
     * @param batchValueCache Value cache output [batchSize, kvDim]
     * @return Tuple2 containing TaskGraph and synchronized GridScheduler
     */
    public Tuple2<ImmutableTaskGraph, GridScheduler> setupVLMTornadoForwardPlanForLayer(
            int layer,
            FloatArray batchInput,
            FloatArray keyWeights,
            FloatArray valueWeights,
            FloatArray batchKeyCache,
            FloatArray batchValueCache) throws Exception {
        
        GridScheduler gridScheduler = setupVLMGridSchedulerForLayer(layer);
        KernelContext context = new KernelContext();
        
        // Create TaskGraph for the specific layer's VLM processing
        TaskGraph vlmLayerGraph = TornadoVMSafeInitializer.createTaskGraphSafely("layer_" + layer)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, 
                            batchInput, keyWeights, valueWeights)
            .task("vlmBatchKeyProjection",
                  TransformerComputeKernelsLayered::vlmBatchKeyProjection,
                  context, batchInput, keyWeights, batchKeyCache,
                  visionTokens, inputDim, kvDim)
            .task("vlmBatchValueProjection",
                  TransformerComputeKernelsLayered::vlmBatchValueProjection,
                  context, batchInput, valueWeights, batchValueCache,
                  visionTokens, inputDim, kvDim)
            .transferToHost(DataTransferMode.EVERY_EXECUTION,
                          batchKeyCache, batchValueCache);
        
        return new Tuple2<>(vlmLayerGraph.snapshot(), gridScheduler);
    }
    
    /**
     * Creates GridScheduler with VLM-specific WorkerGrid configuration for a specific layer
     * 
     * Follows the proven pattern from working non-VLM implementations.
     * Configures optimal thread mapping for VLM batch processing.
     */
    private GridScheduler setupVLMGridSchedulerForLayer(int layer) {
        GridScheduler vlmScheduler = new GridScheduler();
        
        // VLM Batch Processing WorkerGrid (optimized for RTX 2000 Ada)
        WorkerGrid vlmBatchWorker = new WorkerGrid1D(visionTokens);
        vlmBatchWorker.setGlobalWork(visionTokens * 8, 1, 1);  // 144 vision tokens * 8 threads
        vlmBatchWorker.setLocalWork(8, 1, 1);                  // 8 threads per workgroup
        
        // Map VLM tasks to WorkerGrid for the specific layer (matching TaskGraph.Task expected format)
        vlmScheduler.addWorkerGrid("layer_" + layer + ".vlmBatchKeyProjection", vlmBatchWorker);
        vlmScheduler.addWorkerGrid("layer_" + layer + ".vlmBatchValueProjection", vlmBatchWorker);
        
        return vlmScheduler;
    }
}