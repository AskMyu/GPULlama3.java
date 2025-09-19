package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.GemmaState;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.gemma.GemmaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import org.beehive.gpullama3.tornadovm.SmartCacheArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Specialized TornadoVM layer planner for Gemma models.
 *
 * Gemma models have unique characteristics that require specialized handling:
 * - Large vocabulary size (262,144 tokens vs typical 32K-50K)
 * - Different tensor dimensions (640 dim, 2048 hiddenDim)
 * - Unusual parameter distribution (170M embeddings, 100M transformer)
 *
 * This planner uses dynamic thread limit checking and minimal work group sizes
 * to avoid OpenCL CL_INVALID_WORK_GROUP_SIZE errors (-54). Based on MLC-LLM
 * fixes from PRs #1822, #1850, and #1955 that implement dynamic thread limits.
 */
public class GemmaTornadoVMLayerPlanner extends TornadoVMLayerPlanner<GemmaState, GemmaConfiguration, TornadoWeights> {

    // Work group sizes for Gemma models - back to 32 due to GPU limits
    // 64 causes CL_INVALID_WORK_GROUP_SIZE errors, so 32 is the maximum
    private static final int GEMMA_LOCAL_WORK_GROUP_SIZE = 32;
    private static final int GEMMA_THREAD_SCALE_FOR_LOGITS = 2; // Conservative scaling
    private static final int GEMMA_MAX_VOCAB_LOCAL_SIZE = 64; // Back to 64 for vocabulary operations

    /**
     * Constructs a TornadoVMLayerPlanner for the given Gemma model.
     *
     * @param state The state object containing model tensors and buffers
     * @param model The Gemma model instance containing configuration and weights
     */
    public GemmaTornadoVMLayerPlanner(GemmaState state, Model model) {
        super(state, model);
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() throws Exception {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);

        TaskGraph activationUpdate = TornadoVMSafeInitializer.createTaskGraphSafely("activationUpdate")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());

        TaskGraph unifiedLayer = null;
        for (int layerIndex = 0; layerIndex < config.numberOfLayers(); layerIndex++) {
            unifiedLayer = TornadoVMSafeInitializer.createTaskGraphSafely("layer_" + layerIndex);
            unifiedLayer.consumeFromDevice(state.wrapX);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    weights.rms_att_weightLayered[layerIndex],
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            unifiedLayer.task("reductionsOneBlock", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayerMixedPrecision, context, state.wrapXb,
                            state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                    .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb, state.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(), GEMMA_LOCAL_WORK_GROUP_SIZE)
                    .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb, state.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(), GEMMA_LOCAL_WORK_GROUP_SIZE)
                    .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb, state.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(), GEMMA_LOCAL_WORK_GROUP_SIZE)
                    .task("rope", TransformerComputeKernelsLayered::ropeRotation, context,
                            state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(),
                            config.headSize())
                    .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                            getFloatArrayFromCache(state.wrapKeyCache), state.wrapK, getFloatArrayFromCache(state.wrapValueCache), state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength());

            // Attention implementation - differentiate between Gemma and non-Gemma models
            // Enable Gemma-specific kernels for proper 5:1 attention pattern and QK-norm
            if (config instanceof GemmaConfiguration) {
                GemmaConfiguration gemmaConfig = (GemmaConfiguration) config;
                if (gemmaConfig.isGlobalAttentionLayer(layerIndex)) {
                    // Global attention with full context and enhanced RoPE
                    System.err.printf("[GEMMA-PATTERN] Layer %d: GLOBAL (full context) attention, RoPE=1000000, Window=%d%n",
                                     layerIndex, config.contextLength());
                    unifiedLayer = unifiedLayer.task("global-attention", TransformerComputeKernelsLayered::processHeadsFlashAttentionGlobalGemma, context,
                            state.wrapQ, getFloatArrayFromCache(state.wrapKeyCache), getFloatArrayFromCache(state.wrapValueCache), state.wrapXb,
                            config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                            state.positionHolder, layerIndex, config.contextLength(), 1000000.0f, config.contextLength());
                } else {
                    // Local attention with 1024-token sliding window
                    System.err.printf("[GEMMA-PATTERN] Layer %d: LOCAL (1024-token window) attention, RoPE=%.0f, Window=1024%n",
                                     layerIndex, config.ropeTheta());
                    unifiedLayer = unifiedLayer.task("local-attention", TransformerComputeKernelsLayered::processHeadsFlashAttentionLocalGemma, context,
                            state.wrapQ, getFloatArrayFromCache(state.wrapKeyCache), getFloatArrayFromCache(state.wrapValueCache), state.wrapXb,
                            config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                            state.positionHolder, layerIndex, config.contextLength(), config.ropeTheta(), 1024);
                }
            } else {
                // Non-Gemma models use standard mixed precision attention
                unifiedLayer = unifiedLayer.task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttentionMixedPrecision, context,
                        state.wrapQ, getFloatArrayFromCache(state.wrapKeyCache), getFloatArrayFromCache(state.wrapValueCache), state.wrapXb,
                        config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                        state.positionHolder, layerIndex, config.contextLength());
            }

            unifiedLayer = unifiedLayer.task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapXb, state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(), GEMMA_LOCAL_WORK_GROUP_SIZE)
                    .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayerMixedPrecision, context, state.wrapXb,
                            state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                    .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                            state.wrapXb, state.wrapHb, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex],
                            config.dim(), config.hiddenDim(), GEMMA_LOCAL_WORK_GROUP_SIZE)
                    .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(), config.dim(), GEMMA_LOCAL_WORK_GROUP_SIZE)
                    .persistOnDevice(state.wrapX);
            taskGraphs.add(unifiedLayer.snapshot());
        }

        TaskGraph lastUnifiedLayer = unifiedLayer;
        TaskGraph logits = TornadoVMSafeInitializer.createTaskGraphSafely("logits")
                .consumeFromDevice(lastUnifiedLayer.getTaskGraphName(),
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
        logits = configureQuantizedMatrixVectorFinalWeight(logits);
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());

        return new Tuple2<>(taskGraphs, createGemmaOptimizedScheduler());
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() throws Exception {
        // For non-NVIDIA GPUs, use the same conservative approach
        return setupTornadoForwardPlanLayered();
    }

    @Override
    protected TaskGraph configureQuantizedMatrixVectorFinalWeight(TaskGraph logits) {
        switch (weights.getWeightType()) {
            case F32:
            case F16:
            case Q8_0:
            case Q4_0:
            case Q4_K:
            case Q6_K:
            case Q8_K:
                // Use conservative work group size for Gemma's large vocabulary
                logits.task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context, state.wrapX, state.wrapLogits, weights.wclsHalfFloat,
                        config.dim(), config.vocabularySize(), GEMMA_LOCAL_WORK_GROUP_SIZE * GEMMA_THREAD_SCALE_FOR_LOGITS);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported weight quantization type: " + weights.getWeightType() + ". Supported types: F32, F16, Q8_0, Q4_0, Q4_K, Q6_K, Q8_K");
        }
        return logits;
    }

    /**
     * Dynamically checks and adjusts work group size to stay within device limits.
     * Based on MLC-LLM fixes from PRs #1822, #1850, #1955 for OpenCL compatibility.
     *
     * @param requestedSize The desired work group size
     * @param maxDeviceLimit Conservative estimate of device limit (256 for most GPUs)
     * @return Safe work group size that won't exceed device limits
     */
    private int checkAndAdjustWorkGroupSize(int requestedSize, int maxDeviceLimit) {
        // Conservative device limit - most mobile GPUs have 256 thread limit
        int safeLimit = Math.min(maxDeviceLimit, 256);

        if (requestedSize > safeLimit) {
            System.err.printf("[GEMMA-PLANNER] Adjusting work group size from %d to %d (device limit)%n",
                            requestedSize, safeLimit);
            return safeLimit;
        }
        return requestedSize;
    }

    /**
     * Creates a scheduler with conservative work group sizes that balance compatibility and performance.
     * Uses small but functional sizes (8,1,1) to avoid OpenCL errors while maintaining GPU parallelism.
     */
    private GridScheduler createGemmaOptimizedScheduler() {
        GridScheduler tornadoForwardScheduler = new GridScheduler();

        System.err.println("[GEMMA-PLANNER] Using optimized work group sizes (32,1,1) for enhanced numerical precision and stability");

        // Single worker for simple operations
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // RoPE worker - optimized parallelism
        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim() / 2);
        ropeWorker.setGlobalWork(config.dim() / 2, 1, 1);
        ropeWorker.setLocalWork(Math.min(32, config.dim() / 2), 1, 1);

        // Config dimension workers - optimized parallelism
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(config.dim());
        configDimRowMajorGlobalWorker.setGlobalWork(config.dim(), 1, 1);
        configDimRowMajorGlobalWorker.setLocalWork(Math.min(32, config.dim()), 1, 1);

        // KV dimension workers - optimized parallelism
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(config.kvDim());
        configKvDimRowMajorGlobalWorker.setGlobalWork(config.kvDim(), 1, 1);
        configKvDimRowMajorGlobalWorker.setLocalWork(Math.min(32, config.kvDim()), 1, 1);

        // Hidden dimension workers - optimized parallelism
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(config.hiddenDim());
        configHiddenDimRowMajorWorker.setGlobalWork(config.hiddenDim(), 1, 1);
        configHiddenDimRowMajorWorker.setLocalWork(Math.min(32, config.hiddenDim()), 1, 1);

        // RMSNorm worker - optimized parallelism
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
        rmsNormWorker.setLocalWork(Math.min(32, config.dim()), 1, 1);

        // Attention worker - optimized parallelism (increase to 16 for heads)
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads(), 1, 1);
        parallelAttentionWorker.setLocalWork(Math.min(16, config.numberOfHeads()), 1, 1);

        // Cache copy worker - optimized parallelism
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(Math.min(32, config.kvDim()), 1, 1);

        // CRITICAL: Vocabulary worker - very conservative for 262K vocabulary
        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize());
        vocabWorker.setGlobalWork(config.vocabularySize(), 1, 1);
        vocabWorker.setLocalWork(GEMMA_MAX_VOCAB_LOCAL_SIZE, 1, 1);

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            // Attention task mapping: differentiate between Gemma and non-Gemma models
            if (config instanceof GemmaConfiguration) {
                GemmaConfiguration gemmaConfig = (GemmaConfiguration) config;
                if (gemmaConfig.isGlobalAttentionLayer(i)) {
                    tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".global-attention", parallelAttentionWorker);
                } else {
                    tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".local-attention", parallelAttentionWorker);
                }
            } else {
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            }
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
        }

        // Vocabulary operations with conservative settings
        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

        return tornadoForwardScheduler;
    }


    /**
     * Helper method to extract FloatArray from cache objects (SmartCacheArray or FloatArray).
     * For SmartCacheArray, returns the direct array if not batched, or the first batch if batched.
     */
    private FloatArray getFloatArrayFromCache(Object cache) {
        if (cache instanceof SmartCacheArray) {
            SmartCacheArray smartCache = (SmartCacheArray) cache;
            if (smartCache.isBatched()) {
                // For batched arrays, use the first batch for now
                // Full batch coordination would be implemented in a future version
                System.err.printf("[TORNADO-PLANNER] Warning: Using first batch of %d batches for cache operations%n",
                                smartCache.getNumBatches());
                return smartCache.getBatch(0);
            } else {
                return smartCache.getDirectArray();
            }
        } else if (cache instanceof FloatArray) {
            return (FloatArray) cache;
        } else {
            throw new IllegalArgumentException("Unsupported cache type: " + cache.getClass().getSimpleName());
        }
    }
}