package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.List;

/**
 * VLM-specific TornadoVM Master Plan for Single Layer Processing
 * 
 * Provides proper TaskGraph/GridScheduler synchronization for VLM batch processing
 * following the proven TornadoVMMasterPlan architectural pattern.
 * 
 * This class solves the "GridScheduler Name not registered in any task-graph" error
 * by ensuring TaskGraph and GridScheduler are created and synchronized together,
 * rather than independently as in the previous approach.
 */
public class VLMTornadoVMMasterPlan {
    
    private final Configuration config;
    private final GridScheduler gridScheduler;
    private final TornadoExecutionPlan executionPlan;
    private final ImmutableTaskGraph taskGraph;
    private final int targetLayer;
    
    /**
     * Creates VLM master plan with synchronized TaskGraph and GridScheduler for single layer
     * 
     * @param config Model configuration
     * @param layer Target layer to process
     * @param visionTokens Number of vision tokens (144)
     * @param batchInput Vision embeddings
     * @param keyWeights Key projection weights 
     * @param valueWeights Value projection weights
     * @param batchKeyCache Key cache output
     * @param batchValueCache Value cache output
     */
    public VLMTornadoVMMasterPlan(Configuration config, int layer, int visionTokens,
                                  FloatArray batchInput,
                                  FloatArray keyWeights,
                                  FloatArray valueWeights,
                                  FloatArray batchKeyCache,
                                  FloatArray batchValueCache) throws Exception {
        
        this.config = config;
        this.targetLayer = layer;
        
        // Create VLM planner and get synchronized TaskGraph + GridScheduler
        VLMTornadoVMLayerPlanner vlmPlanner = new VLMTornadoVMLayerPlanner(config, visionTokens);
        Tuple2<ImmutableTaskGraph, GridScheduler> vlmPlan = 
            vlmPlanner.setupVLMTornadoForwardPlanForLayer(layer, batchInput, keyWeights, valueWeights,
                                                         batchKeyCache, batchValueCache);
        
        this.taskGraph = vlmPlan.getFirst();
        this.gridScheduler = vlmPlan.getSecond();
        
        // Create execution plan with synchronized TaskGraph and GridScheduler
        this.executionPlan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraph);
        this.executionPlan.withGridScheduler(gridScheduler);
    }
    
    /**
     * Execute VLM batch processing for the configured layer
     */
    public void execute() {
        executionPlan.execute();
    }
    
    /**
     * Get the GridScheduler for debugging purposes
     */
    public GridScheduler getGridScheduler() {
        return gridScheduler;
    }
    
    /**
     * Get the TaskGraph for debugging purposes
     */
    public ImmutableTaskGraph getTaskGraph() {
        return taskGraph;
    }
    
    /**
     * Get the target layer this plan was created for
     */
    public int getTargetLayer() {
        return targetLayer;
    }
}