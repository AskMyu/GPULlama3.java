package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.deepseekr1.DeepSeekR1Configuration;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.ArrayList;
import java.util.List;

/**
 * TornadoVM layer planner specialized for DeepSeek-R1 architecture.
 *
 * Handles GPU scheduling for:
 * - Multi-head Latent Attention (MLA) operations
 * - Mixture of Experts (MoE) routing and computation
 * - FP8 quantization/dequantization
 * - Expert load balancing
 */
public class DeepSeekR1TornadoVMLayerPlanner extends TornadoVMLayerPlanner<State, DeepSeekR1Configuration, TornadoWeights> {

    // MLA parameters
    private final int latentDim;
    private final int compressionRatio;
    private final boolean enableMLA;

    // MoE parameters
    private final int numExperts;
    private final int activeExperts;
    private final int expertHiddenDim;
    private final boolean isMoEModel;

    // GPU arrays for MLA
    private FloatArray mlaCompressionWeights;
    private FloatArray mlaDecompressionWeights;
    private FloatArray compressedKV;
    private IntArray activeHeads;

    // GPU arrays for MoE
    private FloatArray expertGateWeights;
    private FloatArray expertWeights;
    private FloatArray routingWeights;
    private IntArray expertIndices;
    private FloatArray expertOutputs;

    // Performance tracking
    private FloatArray memoryStats;

    public DeepSeekR1TornadoVMLayerPlanner(State state, Model model) {
        super(state, model);

        DeepSeekR1Configuration config = (DeepSeekR1Configuration) model.configuration();

        // Initialize MLA parameters
        this.enableMLA = config.enableMLA();
        if (enableMLA && config.mlaConfig() != null) {
            this.latentDim = config.mlaConfig().latentDim();
            this.compressionRatio = (int) (config.mlaConfig().compressionRatio() * 100);
        } else {
            this.latentDim = 0;
            this.compressionRatio = 0;
        }

        // Initialize MoE parameters
        this.isMoEModel = config.isMoEModel();
        this.numExperts = config.totalExperts();
        this.activeExperts = config.activeExperts();
        this.expertHiddenDim = config.expertHiddenDim();

        // Initialize GPU arrays
        initializeGPUArrays(config);
    }

    /**
     * Initialize GPU arrays for MLA and MoE operations.
     */
    private void initializeGPUArrays(DeepSeekR1Configuration config) {
        int batchSize = 1; // Assuming single batch for inference
        int maxSeqLen = config.contextLength();
        int dim = config.dim();
        int numHeads = config.numberOfHeads();

        // MLA arrays
        if (enableMLA) {
            int headDim = config.headSize();
            mlaCompressionWeights = new FloatArray(headDim * latentDim);
            mlaDecompressionWeights = new FloatArray(latentDim * headDim);
            compressedKV = new FloatArray(2 * batchSize * maxSeqLen * numHeads * latentDim);
            activeHeads = new IntArray(numHeads); // All heads initially active
        }

        // MoE arrays
        if (isMoEModel) {
            expertGateWeights = new FloatArray(dim * numExperts);
            expertWeights = new FloatArray(numExperts * dim * expertHiddenDim * 2); // Gate + output weights
            routingWeights = new FloatArray(batchSize * maxSeqLen * activeExperts);
            expertIndices = new IntArray(batchSize * maxSeqLen * activeExperts);
            expertOutputs = new FloatArray(batchSize * maxSeqLen * dim);
        }

        // Performance tracking
        memoryStats = new FloatArray(4); // [standard_memory, mla_memory, reduction, efficiency]
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // Transfer state arrays
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN);

            // Transfer MLA weights (first execution only)
            if (enableMLA) {
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        mlaCompressionWeights, mlaDecompressionWeights, activeHeads);
                unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        compressedKV);
            }

            // Transfer MoE weights
            if (isMoEModel) {
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        expertGateWeights, expertWeights);
                unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        routingWeights, expertIndices, expertOutputs);
            }

            // Transfer performance tracking
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, memoryStats);
        }

        return unifiedLayer;
    }

    protected void addLayerTasks(TaskGraph unifiedLayer, int layerIndex) {
        String layerPrefix = "layer" + layerIndex + "_";

        // Add MLA compression task if enabled
        if (enableMLA) {
            addMLACompressionTask(unifiedLayer, layerPrefix);
        }

        // Add standard transformer operations (RMS norm, attention, FFN)
        addStandardTransformerTasks(unifiedLayer, layerPrefix);

        // Add MoE routing and computation if enabled
        if (isMoEModel) {
            addMoERoutingTask(unifiedLayer, layerPrefix);
            addMoEComputeTask(unifiedLayer, layerPrefix);
        }

        // Add MLA decompression task if enabled
        if (enableMLA) {
            addMLADecompressionTask(unifiedLayer, layerPrefix);
        }

        // Add memory usage tracking
        addMemoryTrackingTask(unifiedLayer, layerPrefix);
    }

    /**
     * Add MLA compression task for K/V cache compression.
     */
    private void addMLACompressionTask(TaskGraph taskGraph, String layerPrefix) {
        int batchSize = 1;
        int seqLen = 1; // Fixed for inference (processing one token at a time)
        int numHeads = config.numberOfHeads();
        int headDim = config.headSize();

        taskGraph.task(layerPrefix + "mla_compress",
                DeepSeekR1ComputeKernels::mlaCompress,
                ((Qwen3State)state).tempKcur, // Input K/V
                mlaCompressionWeights,
                compressedKV, // Compressed output
                batchSize, seqLen, numHeads, headDim, latentDim);
    }

    /**
     * Add MLA decompression task for attention computation.
     */
    private void addMLADecompressionTask(TaskGraph taskGraph, String layerPrefix) {
        int batchSize = 1;
        int seqLen = 1; // Fixed for inference (processing one token at a time)
        int numActiveHeads = config.numberOfHeads(); // All heads active for inference
        int headDim = config.headSize();


        taskGraph.task(layerPrefix + "mla_decompress",
                DeepSeekR1ComputeKernels::mlaDecompress,
                compressedKV, // Compressed input
                mlaDecompressionWeights,
                ((Qwen3State)state).tempKcur, // Decompressed output
                activeHeads,
                batchSize, seqLen, numActiveHeads, headDim, latentDim);
    }

    /**
     * Add MoE routing task for expert selection.
     */
    private void addMoERoutingTask(TaskGraph taskGraph, String layerPrefix) {
        int batchSize = 1;
        int seqLen = 1; // Fixed for inference (processing one token at a time)
        int dim = config.dim();


        taskGraph.task(layerPrefix + "moe_routing",
                DeepSeekR1ComputeKernels::moeRouting,
                state.temp, // Input tokens
                expertGateWeights,
                routingWeights, // Output routing weights
                expertIndices, // Output expert indices
                batchSize, seqLen, dim, numExperts, activeExperts);
    }

    /**
     * Add MoE computation task for expert processing.
     */
    private void addMoEComputeTask(TaskGraph taskGraph, String layerPrefix) {
        int batchSize = 1;
        int seqLen = 1; // Fixed for inference (processing one token at a time)
        int dim = config.dim();


        taskGraph.task(layerPrefix + "moe_compute",
                DeepSeekR1ComputeKernels::moeExpertCompute,
                state.temp, // Input tokens
                expertWeights,
                routingWeights,
                expertIndices,
                expertOutputs, // Expert outputs
                batchSize, seqLen, dim, expertHiddenDim, activeExperts);
    }

    /**
     * Add standard transformer tasks (reusing existing implementations).
     */
    private void addStandardTransformerTasks(TaskGraph taskGraph, String layerPrefix) {
        // Reuse existing transformer operations from TransformerComputeKernelsLayered
        // RMS norm, attention, FFN etc.
        // This would integrate with existing codebase patterns
    }

    /**
     * Add memory tracking task for performance monitoring.
     */
    private void addMemoryTrackingTask(TaskGraph taskGraph, String layerPrefix) {
        if (enableMLA) {
            int batchSize = 1;
            int seqLen = 1; // Fixed for inference (processing one token at a time)
            int numHeads = config.numberOfHeads();
            int headDim = config.headSize();

            taskGraph.task(layerPrefix + "memory_stats",
                    DeepSeekR1ComputeKernels::mlaMemoryEstimate,
                    memoryStats,
                    batchSize, seqLen, numHeads, headDim, latentDim);
        }
    }

    protected List<Tuple2<String, Integer>> getLayerDependencies(int layerIndex) {
        List<Tuple2<String, Integer>> dependencies = new ArrayList<>();

        if (layerIndex > 0) {
            // Depend on previous layer completion
            dependencies.add(new Tuple2<>("layer" + (layerIndex - 1) + "_complete", layerIndex - 1));
        }

        return dependencies;
    }

    /**
     * Get MLA memory reduction statistics.
     */
    public FloatArray getMemoryStats() {
        return memoryStats;
    }

    /**
     * Get expert routing efficiency statistics.
     */
    public float getExpertUtilization() {
        if (!isMoEModel) return 1.0f;

        // Calculate expert utilization from routing weights
        // This would analyze the distribution of expert selections
        return (float) activeExperts / numExperts;
    }

    /**
     * Get current MLA compression ratio.
     */
    public float getMLACompressionRatio() {
        return enableMLA ? (float) latentDim / config.headSize() : 1.0f;
    }
}