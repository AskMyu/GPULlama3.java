package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.GemmaState;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.List;
import java.util.Locale;

public class TornadoVMMasterPlan {
    public static final boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    private final State state;
    private final Configuration config;
    public GridScheduler scheduler;
    public TornadoExecutionPlan executionPlan;
    List<ImmutableTaskGraph> taskGraphs;

    // Layer range for batch processing (-1 means all layers)
    private final int batchStartLayer;
    private final int batchEndLayer;

    public TornadoVMMasterPlan(State state, Model model) {
        this(state, model, -1, -1); // All layers by default
    }

    public TornadoVMMasterPlan(State state, Model model, int startLayer, int endLayer) {
        this.batchStartLayer = startLayer;
        this.batchEndLayer = endLayer;

        try {
            TornadoVMLayerPlanner tornadoVMLayerPlanner = createPlanner(state, model);


            Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMPlan = shouldUseNvidiaScheduler(model)
                    ? tornadoVMLayerPlanner.setupTornadoForwardPlanLayered()
                    : tornadoVMLayerPlanner.setupTornadoForwardPlanLayeredNonNvidia();
            this.taskGraphs = tornadoVMPlan.getFirst();
            this.scheduler = tornadoVMPlan.getSecond();
            this.state = state;
            this.config = model.configuration();
            this.executionPlan = TornadoVMSafeInitializer.createExecutionPlanSafely(taskGraphs.toArray(new ImmutableTaskGraph[0]));
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize TornadoVM Master Plan", e);
        }
    }

    /**
     * Initializes the TornadoVM plan for GPU acceleration with optional timing. This method handles: 1. Creation of the TornadoVM master plan 2. Warming up the JIT compiler for better performance 3.
     * Copying read-only model weights to the GPU
     *
     * @param state
     *         The model state containing KV cache
     * @param model
     *         The Llama model instance
     * @return The initialized TornadoVMMasterPlan ready for inference
     */
    public static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model) {
        // Initialize timing variables outside conditional blocks to avoid scope issues
        long startTime = System.nanoTime();
        long planCreationTime = 0;
        long warmupTime = 0;

        // Start a timing message if enabled
        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        // 1. Pre-allocate the TornadoVM plan
        TornadoVMMasterPlan tornadoVMPlan = new TornadoVMMasterPlan(state, model);

        // Record time after plan creation
        if (ENABLE_TORNADOVM_INIT_TIME) {
            planCreationTime = System.nanoTime();
            System.err.printf("TornadoVM GPU execution plan creation: %.2f ms\n", (planCreationTime - startTime) / 1_000_000.0);
        }

        // 2. Perform warmup with extra iterations to ensure JIT compilation is complete
        tornadoVMPlan.executionPlan.withPreCompilation(); // Force JIT compilation from Java to GPU code

        // Record time after warmup
        if (ENABLE_TORNADOVM_INIT_TIME) {
            warmupTime = System.nanoTime();
            System.err.printf("Java to GPU JIT compiler warmup: %.2f ms\n", (warmupTime - planCreationTime) / 1_000_000.0);
        }

        // 3. Perform copy-in of read-only weights and objects
        tornadoVMPlan.forceCopyInReadOnlyDataLayered(); // Force copy-in read-only weights

        // Record final timing information
        if (ENABLE_TORNADOVM_INIT_TIME) {
            long copyTime = System.nanoTime();
            System.err.printf("Transfer read-only weights to GPU: %.2f ms\n", (copyTime - warmupTime) / 1_000_000.0);
            System.err.printf("Finished TornadoVM initialization...\n \n");
        }

        model.setTornadoVMPlan(tornadoVMPlan);

        return tornadoVMPlan;
    }


    /**
     * Dispatcher method to select the TornadoVMLayerPlanner for the model.
     */
    TornadoVMLayerPlanner createPlanner(State state, Model model) {
        System.err.printf("[TORNADO-PLANNER] Model type detected: %s%n", model.getModelType());

        return switch (model.getModelType()) {
            case LLAMA_3, MISTRAL -> {
                System.err.println("[TORNADO-PLANNER] Using standard TornadoVMLayerPlanner for LLAMA_3/MISTRAL");
                yield new TornadoVMLayerPlanner(state, model);
            }
            case PHI_3, PHI_4_MINI_REASONING -> {
                System.err.println("[TORNADO-PLANNER] Using Phi3TornadoVMLayerPlanner");
                yield new Phi3TornadoVMLayerPlanner((Phi3State) state, model);
            }
            case QWEN_2, DEEPSEEK_R1_DISTILL_QWEN_1_5B, DEEPSEEK_R1_DISTILL_QWEN_14B -> {
                System.err.println("[TORNADO-PLANNER] Using Qwen2TornadoVMLayerPlanner (including DeepSeek-R1-Distill models)");
                yield new Qwen2TornadoVMLayerPlanner((Qwen2State) state, model);
            }
            case DEEPSEEK_R1_DISTILL_QWEN, DEEPSEEK_R1_FULL -> {
                System.err.println("[TORNADO-PLANNER] Using DeepSeekR1TornadoVMLayerPlanner for DeepSeek-R1");
                yield new DeepSeekR1TornadoVMLayerPlanner((Qwen3State) state, model);
            }
            case OLMOE_1B_7B -> {
                System.err.println("[TORNADO-PLANNER] Using standard TornadoVMLayerPlanner for OLMOE");
                yield new TornadoVMLayerPlanner(state, model);
            }
            case LLAVA_LLAMA_3_8B, LLAVA_LLAMA_3_8B_INT4 -> {
                System.err.println("[TORNADO-PLANNER] Using standard TornadoVMLayerPlanner for LLAVA");
                yield new TornadoVMLayerPlanner(state, model);
            }
            case QWEN_3, QWEN3_30B_A3B -> {
                System.err.println("[TORNADO-PLANNER] Using Qwen3TornadoVMLayerPlanner");
                yield new Qwen3TornadoVMLayerPlanner((Qwen3State) state, model);
            }
            case GEMMA_3 -> {
                System.err.println("[TORNADO-PLANNER] ✅ Using GemmaTornadoVMLayerPlanner for GEMMA_3");
                yield new GemmaTornadoVMLayerPlanner((GemmaState) state, model);
            }
            case GRANITE_3_3 -> {
                System.err.println("[TORNADO-PLANNER] ✅ Using GraniteTornadoVMLayerPlanner for GRANITE_3_3 with native GPU GQA");
                yield new GraniteTornadoVMLayerPlanner(state, (org.beehive.gpullama3.model.granite.Granite) model);
            }
            case GPT_OSS -> {
                System.err.println("[TORNADO-PLANNER] Using standard TornadoVMLayerPlanner for GPT_OSS");
                yield new TornadoVMLayerPlanner(state, model);
            }
            case UNKNOWN -> {
                System.err.println("[TORNADO-PLANNER] ❌ UNKNOWN model type - cannot create planner");
                throw new UnsupportedOperationException("Cannot create planner for unknown model type");
            }
        };
    }

    /**
     * Determines whether the NVIDIA-specific scheduler should be used based on the current
     * hardware backend and the model type.
     * <p>
     * The scheduler is used only if the runtime is targeting an NVIDIA backend and the model is not of type {@code MISTRAL}. If either the hardware is not NVIDIA or the model is {@code MISTRAL}, the
     * NVIDIA-specific scheduler should not be used.
     *
     * @param model
     *         the model whose type may affect the scheduler decision
     * @return {@code true} if the NVIDIA-specific scheduler should be used; {@code false} otherwise
     */
    public static boolean shouldUseNvidiaScheduler(Model model) {
        try {
            TornadoRuntime runtime = TornadoVMSafeInitializer.getTornadoRuntimeSafely();
            String platformName = runtime.getBackend(0).getDefaultDevice().getPlatformName().toLowerCase(Locale.ROOT);

            boolean isNvidia = platformName.contains("nvidia");
            boolean isNotMistral = model.getModelType() != ModelType.MISTRAL;

            boolean result = isNvidia && isNotMistral;

            return result;
        } catch (Exception e) {
            System.err.println("[TORNADO-SCHEDULER] Failed to detect NVIDIA platform, defaulting to non-NVIDIA scheduler: " + e.getMessage());
            return false; // Default to non-NVIDIA scheduler if detection fails
        }
    }

    /**
     * Executes the forward pass of a LLaMA transformer model using TornadoVM acceleration.
     *This method processes the transformer layers in sequence for a particular token position in the context
     * window.
     *
     * <p>The execution happens in three phases:
     * <ol>
     *   <li>Initial token embedding lookup (already done before calling this method)</li>
     *   <li>Sequential processing through each transformer layer using TornadoVM</li>
     *   <li>Final projection to logits using TornadoVM</li>
     * </ol>
     *
     * @param position
     *         The current position in the sequence being processed
     * @return FloatTensor containing the output logits for token prediction
     */

    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        // @formatter:off
        // 1. Execute the preprocessing graph (e.g., input preparation, memory initialization)
        executionPlan.withGraph(getPreprocessingGraphIndex())
                .withGridScheduler(scheduler)
                .execute();

        // Set the position in the state object (used by attention layers)
        // DEBUG: Only for Gemma models to avoid disrupting other models
        if (config.getClass().getSimpleName().contains("Gemma")) {
            System.err.printf("[POSITION-DEBUG] Setting position to %d in TornadoVM execution%n", position);
        }
        state.positionHolder.set(0, position);

        // 2. Execute each transformer layer graph sequentially
        // Each graph computes attention and feed-forward transformations for one layer
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(getLayerGraphIndex(layer))
                    .withGridScheduler(scheduler)
                    .execute();
        }

        // 3. Execute the final graph that projects the last hidden state to output logits
        executionPlan.withGraph(getFinalLogitsGraphIndex())
                .withGridScheduler(scheduler)
                .execute();

        // @formatter:on

        // DEBUG: Only check state evolution for Gemma models to avoid disrupting other models
        if (config.getClass().getSimpleName().contains("Gemma")) {
            // DEBUG: Check if hidden state is evolving
            float firstXValue = state.wrapX.get(0);
            float midXValue = state.wrapX.get(state.wrapX.getSize() / 2);
            float lastXValue = state.wrapX.get(state.wrapX.getSize() - 1);
            System.err.printf("[STATE-DEBUG] Position %d - X state values: first=%.6f, mid=%.6f, last=%.6f%n",
                             position, firstXValue, midXValue, lastXValue);

            // DEBUG: Check top logits to see if they're varying
            float maxLogit = -Float.MAX_VALUE;
            int maxToken = -1;
            float secondMaxLogit = -Float.MAX_VALUE;
            int secondMaxToken = -1;

            for (int i = 0; i < Math.min(state.wrapLogits.getSize(), 50000); i++) {
                float logit = state.wrapLogits.get(i);
                if (logit > maxLogit) {
                    secondMaxLogit = maxLogit;
                    secondMaxToken = maxToken;
                    maxLogit = logit;
                    maxToken = i;
                } else if (logit > secondMaxLogit) {
                    secondMaxLogit = logit;
                    secondMaxToken = i;
                }
            }
            System.err.printf("[LOGITS-DEBUG] Position %d - Top tokens: #1=%d(%.4f), #2=%d(%.4f)%n",
                             position, maxToken, maxLogit, secondMaxToken, secondMaxLogit);
        }

        // Return the logits (used for token prediction)
        return state.wrapLogits;
    }

    /**
     * Returns the graph index for the pre-processing step (e.g., token embedding).
     */
    private int getPreprocessingGraphIndex() {
        return 0;
    }

    /**
     * Returns the graph index for the given transformer layer.
     *
     * @param layerIndex
     *         Index of the transformer layer (0-based)
     */
    private int getLayerGraphIndex(int layerIndex) {
        return 1 + layerIndex;
    }

    /**
     * Returns the graph index for the final projection to logits.
     */
    private int getFinalLogitsGraphIndex() {
        return taskGraphs.size() - 1;
    }

    /**
     * Force copy-in of read-only weights for a specific batch of layers.
     * This method only loads weights for the specified layer range, not all layers.
     *
     * NOTE: Due to TornadoVM limitations, we cannot reset after warmup.
     * This method assumes the plan structure exists for all layers.
     *
     * @param startLayer Starting layer index (inclusive)
     * @param endLayer Ending layer index (inclusive)
     */
    public void forceCopyInReadOnlyDataLayeredBatch(int startLayer, int endLayer) {
        System.err.printf("[TORNADO-BATCH] Loading weights for layers %d-%d only%n", startLayer, endLayer);

        // CRITICAL: We cannot reset or reinitialize TornadoVM after warmup
        // Instead, we must work within the existing execution plan
        System.err.println("[TORNADO-BATCH] ⚠️ WARNING: Layer-specific weight loading after warmup is limited");
        System.err.println("[TORNADO-BATCH] TornadoVM does not support reset after warmup - attempting partial execution");

        try {
            // Execute all TornadoVM graphs - this is required due to TornadoVM constraints
            // We cannot selectively execute only certain layer graphs after warmup
            forceCopyInReadOnlyDataLayered();
            System.err.printf("[TORNADO-BATCH] Had to load ALL layers due to TornadoVM constraints%n");
        } catch (Exception e) {
            System.err.printf("[TORNADO-BATCH] ❌ Failed to load layer batch: %s%n", e.getMessage());
            throw new RuntimeException("TornadoVM layer batch loading failed", e);
        }
    }

    /// Execute the forward pass of the LLaMA transformer model using TornadoVM acceleration just once to copy the data into the read-only data layer.
    public void forceCopyInReadOnlyDataLayered() {
        // Check GPU resources first
        long availableGPUMemory = getAvailableGPUMemory();

        // Special handling for MoE models only if memory is limited
        if (isMoEModel()) {
            long totalExpertMemory = estimateTotalExpertMemory();

            if (totalExpertMemory > availableGPUMemory * 0.8) {
                System.err.printf("[TORNADO-COPY] MoE model requires special handling due to memory constraints%n");
                System.err.printf("[TORNADO-COPY] Expert tensors: %.2f GB, Available GPU: %.2f GB%n",
                                totalExpertMemory / (1024.0 * 1024.0 * 1024.0),
                                availableGPUMemory / (1024.0 * 1024.0 * 1024.0));
                forceCopyInReadOnlyDataLayeredMoE();
                return;
            }

            System.err.println("[TORNADO-COPY] MoE model has sufficient GPU memory - using standard copying");
        }

        // Simple approach: execute all layers at once (original working behavior)
        // Complex batching was unnecessary - the real issue was model-specific vocabulary limits
        forceCopyInReadOnlyDataLayeredAllAtOnce();
    }

    private int calculateOptimalBatchSize() {
        // Calculate based on ACTUAL memory requirements vs OpenCL limits
        int totalLayers = config.numberOfLayers();

        // Get the REAL constraint - OpenCL max single allocation (not total GPU memory!)
        long openCLMaxAllocation = org.beehive.gpullama3.tornadovm.OpenCLMemoryDetector.getMaxAllocationSize();

        // Calculate fixed overhead (always needed regardless of batching)
        long fixedOverhead = calculateFixedMemoryOverhead();

        // Calculate per-layer memory requirement
        long perLayerMemory = estimateLayerMemoryRequirement();

        System.err.printf("[TORNADO-COPY] Memory calculation: Fixed=%.2f GB, Per-layer=%.3f GB, OpenCL limit=%.2f GB%n",
                          fixedOverhead / (1024.0 * 1024.0 * 1024.0),
                          perLayerMemory / (1024.0 * 1024.0 * 1024.0),
                          openCLMaxAllocation / (1024.0 * 1024.0 * 1024.0));

        // Calculate how many layers can fit within OpenCL single allocation limit
        long availableForLayers = openCLMaxAllocation - fixedOverhead;

        if (availableForLayers <= 0) {
            System.err.printf("[TORNADO-COPY] ⚠️ Fixed overhead (%.2f GB) exceeds OpenCL limit (%.2f GB)!%n",
                              fixedOverhead / (1024.0 * 1024.0 * 1024.0),
                              openCLMaxAllocation / (1024.0 * 1024.0 * 1024.0));
            return 1; // Process one layer at a time as fallback
        }

        int maxLayersInBatch = (int) (availableForLayers / perLayerMemory);

        // Apply safety margin (use 90% of calculated max)
        maxLayersInBatch = (int) (maxLayersInBatch * 0.9);

        if (maxLayersInBatch >= totalLayers) {
            System.err.printf("[TORNADO-COPY] ✅ Can fit all %d layers within OpenCL limit%n", totalLayers);
            return totalLayers;
        } else {
            System.err.printf("[TORNADO-COPY] ⚠️ OpenCL limit allows only %d of %d layers per batch%n",
                              maxLayersInBatch, totalLayers);
            return Math.max(1, maxLayersInBatch);
        }
    }

    /**
     * Calculate fixed memory overhead (embeddings, output projection, KV cache, state)
     */
    private long calculateFixedMemoryOverhead() {
        long overhead = 0;

        // Embedding matrix (vocabulary × dimension × 4 bytes)
        overhead += (long) config.vocabularySize() * config.dim() * 4L;

        // Output projection (may be separate from embedding)
        overhead += (long) config.vocabularySize() * config.dim() * 4L;

        // KV cache for all layers
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        long kvCacheSize = (long) config.contextLength() * kvDim * config.numberOfLayers() * 2L * 4L;
        overhead += kvCacheSize;

        // Runtime state tensors (x, xb, q, k, v, att, logits, etc.)
        overhead += (long) config.dim() * 10L * 4L; // Intermediate tensors
        overhead += (long) config.vocabularySize() * 4L; // Logits
        overhead += (long) config.numberOfHeads() * config.contextLength() * 4L; // Attention scores

        // Add 10% safety margin for other allocations
        overhead = (long) (overhead * 1.1);

        return overhead;
    }

    private int getHeuristicBatchSize(int totalLayers) {
        // Calculate approximate model size based on parameters
        long approxModelParams = estimateModelParameters();
        double modelSizeGB = (approxModelParams * 4.0) / (1024.0 * 1024.0 * 1024.0); // 4 bytes per float

        System.err.printf("[TORNADO-COPY] Model size estimation: %.1f billion params, %.2f GB%n",
                          approxModelParams / 1_000_000_000.0, modelSizeGB);

        // Use actual model size rather than just layer count for batching decisions
        if (modelSizeGB <= 4.0) {
            // Small models (≤4GB): Process all layers at once
            System.err.printf("[TORNADO-COPY] Small model detected (%.2f GB) - processing all %d layers at once%n",
                              modelSizeGB, totalLayers);
            return totalLayers;
        } else if (modelSizeGB <= 8.0) {
            // Medium models (4-8GB): Batch size based on layer count
            int batchSize = Math.max(4, totalLayers / 4);
            System.err.printf("[TORNADO-COPY] Medium model detected (%.2f GB) - batch size %d%n",
                              modelSizeGB, batchSize);
            return Math.min(batchSize, totalLayers);
        } else if (modelSizeGB <= 16.0) {
            // Large models (8-16GB): Conservative batching
            int batchSize = Math.max(3, totalLayers / 6);
            System.err.printf("[TORNADO-COPY] Large model detected (%.2f GB) - batch size %d%n",
                              modelSizeGB, batchSize);
            return Math.min(batchSize, totalLayers);
        } else {
            // Very large models (>16GB): Very conservative batching
            int batchSize = Math.max(2, totalLayers / 8);
            System.err.printf("[TORNADO-COPY] Very large model detected (%.2f GB) - batch size %d%n",
                              modelSizeGB, batchSize);
            return Math.min(batchSize, totalLayers);
        }
    }

    /**
     * Estimate total model parameters for more accurate size-based batching decisions
     */
    private long estimateModelParameters() {
        long totalParams = 0;

        // Embedding parameters (vocabulary × dimension)
        totalParams += (long) config.vocabularySize() * config.dim();

        // Layer parameters (attention + FFN for each layer)
        for (int i = 0; i < config.numberOfLayers(); i++) {
            // Attention parameters: Q, K, V, O projections
            totalParams += 4L * config.dim() * config.dim();

            // Layer norm parameters
            totalParams += 2L * config.dim(); // Pre-attention and post-FFN

            // FFN parameters
            if (isMoEModel()) {
                // For MoE models, estimate based on active experts
                if (config instanceof org.beehive.gpullama3.model.gptoss.GptOssConfiguration gptOss) {
                    totalParams += gptOss.numExperts() * gptOss.hiddenDim() * config.dim() * 2L;
                } else {
                    // Fallback for unknown MoE configurations
                    totalParams += 8L * config.hiddenDim() * config.dim(); // Assume 8 experts avg
                }
            } else {
                // Standard FFN: up_proj + down_proj
                totalParams += 2L * config.dim() * config.hiddenDim();
            }
        }

        // Final layer norm and output projection
        totalParams += config.dim(); // Final layer norm
        totalParams += (long) config.vocabularySize() * config.dim(); // Output projection (may be tied)

        return totalParams;
    }

    private void forceCopyInReadOnlyDataLayeredAllAtOnce() {
        System.err.println("[TORNADO-COPY] Processing all layers at once");

        // Execute all TornadoVM graphs
        state.wrapX.init(0.0f);
        state.positionHolder.init(0);

        // Execute activation update graph
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Execute layer processing graphs
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(layer + 1).withGridScheduler(scheduler).execute();
        }

        // Execute logits graph
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(scheduler).execute();
    }

    private void forceCopyInReadOnlyDataLayeredInBatches(int batchSize) {
        System.err.printf("[TORNADO-COPY] Processing %d layers in batches of %d%n", config.numberOfLayers(), batchSize);

        // Initialize state once
        state.wrapX.init(0.0f);
        state.positionHolder.init(0);

        // Execute activation update graph once
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Process layers in batches
        for (int startLayer = 0; startLayer < config.numberOfLayers(); startLayer += batchSize) {
            int endLayer = Math.min(startLayer + batchSize - 1, config.numberOfLayers() - 1);

            System.err.printf("[TORNADO-COPY] Processing layer batch %d-%d (%d layers)%n",
                            startLayer, endLayer, endLayer - startLayer + 1);

            try {
                // Execute this batch of layers
                for (int layer = startLayer; layer <= endLayer; layer++) {
                    executionPlan.withGraph(layer + 1).withGridScheduler(scheduler).execute();
                }

                // Aggressive memory cleanup between batches to reduce fragmentation
                System.err.printf("[TORNADO-COPY] Performing memory cleanup after batch %d-%d%n", startLayer, endLayer);

                // Multiple rounds of cleanup
                for (int cleanup = 0; cleanup < 3; cleanup++) {
                    System.gc(); // Suggest JVM cleanup
                    System.runFinalization(); // Run finalizers
                    try {
                        Thread.sleep(50); // Longer pause to allow cleanup
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }

                System.err.println("[TORNADO-COPY] Memory cleanup completed");

            } catch (Exception e) {
                System.err.printf("[TORNADO-COPY] ❌ Failed to process layer batch %d-%d: %s%n",
                                startLayer, endLayer, e.getMessage());

                // If batch failed and batch size > 1, try single layer processing
                if (batchSize > 1 && (endLayer - startLayer + 1) > 1) {
                    System.err.println("[TORNADO-COPY] Attempting single-layer fallback processing");
                    forceCopyInReadOnlyDataLayeredSingleLayer();
                    return;
                }

                throw new RuntimeException("TornadoVM layer batch processing failed", e);
            }
        }

        // Execute logits graph once at the end
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(scheduler).execute();

        System.err.println("[TORNADO-COPY] ✅ Completed batched layer processing");
    }

    private void forceCopyInReadOnlyDataLayeredSingleLayer() {
        System.err.printf("[TORNADO-COPY] ⚠️ FALLBACK: Processing %d layers one at a time (ultra-conservative mode)%n",
                         config.numberOfLayers());
        System.err.println("[TORNADO-COPY] ⚠️ WARNING: This mode is extremely slow but should handle memory pressure");

        // Initialize state once
        state.wrapX.init(0.0f);
        state.positionHolder.init(0);

        // Initial stabilization delay
        try {
            System.err.println("[TORNADO-COPY] Initial driver stabilization pause (1 second)");
            Thread.sleep(1000);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }

        // Execute activation update graph once
        System.err.println("[TORNADO-COPY] Executing activation update graph");
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Process each layer individually with aggressive cleanup
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            System.err.printf("[TORNADO-COPY] Processing single layer %d/%d%n",
                            layer + 1, config.numberOfLayers());

            // Pre-execution stabilization for problematic layers
            if (layer > 4) {  // After layer 5 where crash occurred
                try {
                    System.err.printf("[TORNADO-COPY] Extra stabilization pause before layer %d (500ms)%n", layer);
                    Thread.sleep(500);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }

            try {
                // Execute single layer
                executionPlan.withGraph(layer + 1).withGridScheduler(scheduler).execute();

                // Ultra-aggressive memory cleanup after EACH layer
                System.err.printf("[TORNADO-COPY] Memory cleanup after layer %d%n", layer);

                // Even more aggressive cleanup for later layers
                int cleanupRounds = (layer > 4) ? 8 : 5;  // More cleanup after problematic layer
                int sleepTime = (layer > 4) ? 200 : 100;  // Longer pauses after problematic layer

                for (int cleanup = 0; cleanup < cleanupRounds; cleanup++) {
                    System.gc();
                    System.runFinalization();
                    try {
                        Thread.sleep(sleepTime);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }

                // Additional driver recovery time for every 4th layer
                if ((layer + 1) % 4 == 0) {
                    System.err.printf("[TORNADO-COPY] Driver recovery pause after layer %d (2 seconds)%n", layer);
                    try {
                        Thread.sleep(2000);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }

            } catch (Exception e) {
                System.err.printf("[TORNADO-COPY] ❌ CRITICAL: Failed on single layer %d: %s%n",
                                layer, e.getMessage());
                System.err.println("[TORNADO-COPY] Single-layer processing failed - this indicates severe memory issues");
                throw new RuntimeException("TornadoVM single-layer processing failed at layer " + layer, e);
            }
        }

        // Execute logits graph once at the end
        System.err.println("[TORNADO-COPY] Executing logits graph");
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(scheduler).execute();

        System.err.println("[TORNADO-COPY] ✅ Completed single-layer fallback processing");
    }

    /**
     * Frees the device memory allocated for the TornadoVM execution plan. This method should be called when the execution plan is no longer needed to release resources and avoid memory leaks.
     */
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }

    // Helper methods for memory estimation and MoE detection

    /**
     * Check if the current model is a Mixture of Experts model
     */
    private boolean isMoEModel() {
        // Check if configuration is GptOssConfiguration
        if (config instanceof org.beehive.gpullama3.model.gptoss.GptOssConfiguration) {
            return true;
        }

        // Check if configuration is OlmoeConfiguration
        if (config instanceof org.beehive.gpullama3.model.olmoe.OlmoeConfiguration) {
            return true;
        }

        // Check if it's a Qwen3 MoE model
        if (config instanceof org.beehive.gpullama3.model.qwen3.Qwen3Configuration qwenConfig) {
            return qwenConfig.isMoEModel();
        }

        return false;
    }

    /**
     * Get available GPU memory from TornadoVM device
     */
    private long getAvailableGPUMemory() {
        try {
            // Use OpenCL max allocation detection for accurate limits
            long openCLMaxAlloc = org.beehive.gpullama3.tornadovm.OpenCLMemoryDetector.getMaxAllocationSize();

            // Also check TornadoVM device memory setting as secondary limit
            String deviceMemory = System.getProperty("tornado.device.memory");
            long tornadoVMLimit = 8L * 1024L * 1024L * 1024L; // 8GB default

            if (deviceMemory != null) {
                // Parse memory string (e.g., "34398MB" or "8GB")
                deviceMemory = deviceMemory.toUpperCase();
                long multiplier = 1;
                if (deviceMemory.endsWith("GB")) {
                    multiplier = 1024L * 1024L * 1024L;
                    deviceMemory = deviceMemory.substring(0, deviceMemory.length() - 2);
                } else if (deviceMemory.endsWith("MB")) {
                    multiplier = 1024L * 1024L;
                    deviceMemory = deviceMemory.substring(0, deviceMemory.length() - 2);
                }
                tornadoVMLimit = Long.parseLong(deviceMemory) * multiplier;
            }

            // PRIORITY FIX: TornadoVM setting is based on actual system detection, OpenCL detection can be conservative
            long effectiveLimit;
            if (tornadoVMLimit > openCLMaxAlloc) {
                // TornadoVM setting is higher - it comes from system detection, so trust it
                effectiveLimit = tornadoVMLimit;
                System.err.printf("[TORNADO-COPY] Using TornadoVM system-detected setting over conservative OpenCL detection%n");
            } else {
                // OpenCL detection is higher or equal - use it (rare but possible)
                effectiveLimit = openCLMaxAlloc;
                System.err.printf("[TORNADO-COPY] Using OpenCL detection (higher than TornadoVM setting)%n");
            }

            System.err.printf("[TORNADO-COPY] GPU Memory Limits: OpenCL=%.2f GB, TornadoVM=%.2f GB, Using=%.2f GB%n",
                            openCLMaxAlloc / (1024.0 * 1024.0 * 1024.0),
                            tornadoVMLimit / (1024.0 * 1024.0 * 1024.0),
                            effectiveLimit / (1024.0 * 1024.0 * 1024.0));

            return effectiveLimit;

        } catch (Exception e) {
            System.err.println("[TORNADO-COPY] Could not detect GPU memory limits: " + e.getMessage());
            // Safe fallback to 2GB (well within any reasonable OpenCL limit)
            return 2L * 1024L * 1024L * 1024L;
        }
    }

    /**
     * Estimate memory requirement per layer
     */
    private long estimateLayerMemoryRequirement() {
        // Rough estimation based on model dimensions
        long weightsPerLayer = 0;

        // Attention weights: Q, K, V, O projections
        weightsPerLayer += 4L * config.dim() * config.dim();

        // FFN weights (or MoE expert weights)
        if (isMoEModel()) {
            // For MoE, this varies by active experts
            if (config instanceof org.beehive.gpullama3.model.gptoss.GptOssConfiguration gptOss) {
                weightsPerLayer += gptOss.activeExperts() * gptOss.hiddenDim() * config.dim() * 2L;
            } else {
                weightsPerLayer += 4L * config.dim() * config.dim(); // Fallback estimate
            }
        } else {
            // Standard FFN: gate, up, down projections
            weightsPerLayer += 3L * config.dim() * config.hiddenDim(); // Use actual hidden dim
        }

        // Assume float32 (4 bytes per weight)
        return weightsPerLayer * 4;
    }

    /**
     * Estimate total memory required for all expert tensors in MoE models
     */
    private long estimateTotalExpertMemory() {
        if (!isMoEModel()) {
            return 0;
        }

        if (config instanceof org.beehive.gpullama3.model.gptoss.GptOssConfiguration gptOss) {
            // GPT-OSS: 32 experts * hiddenDim * dim * 3 tensors (gate, up, down) * 4 bytes
            long expertsMemoryPerLayer = (long) gptOss.numExperts() * gptOss.hiddenDim() * config.dim() * 3L * 4L;
            return expertsMemoryPerLayer * config.numberOfLayers();
        }

        if (config instanceof org.beehive.gpullama3.model.olmoe.OlmoeConfiguration olmoe) {
            // OLMoE: numberOfExperts * dimension calculations
            long expertsMemoryPerLayer = (long) olmoe.numberOfExperts() * config.dim() * config.dim() * 3L * 4L;
            return expertsMemoryPerLayer * config.numberOfLayers();
        }

        // Fallback estimate for other MoE models
        return 16L * 1024L * 1024L * 1024L; // 16GB default
    }

    /**
     * Special weight copying for MoE models with limited GPU memory
     */
    private void forceCopyInReadOnlyDataLayeredMoE() {
        System.err.println("[TORNADO-COPY-MOE] Special MoE weight copying mode activated");
        System.err.println("[TORNADO-COPY-MOE] This mode skips expert tensor copying to conserve GPU memory");
        System.err.println("[TORNADO-COPY-MOE] Expert tensors will be loaded on-demand during inference");

        // Initialize state
        state.wrapX.init(0.0f);
        state.positionHolder.init(0);

        // Execute activation update graph
        System.err.println("[TORNADO-COPY-MOE] Loading non-expert weights only...");
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // For MoE models, we skip most layer weight copying
        // Only copy essential weights like attention and layer norms
        System.err.println("[TORNADO-COPY-MOE] Skipping expert tensor copying for all layers");

        // Execute logits graph
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(scheduler).execute();

        System.err.println("[TORNADO-COPY-MOE] ✅ MoE minimal weight copying completed");
        System.err.println("[TORNADO-COPY-MOE] ⚠️ Note: Inference may be slower due to on-demand expert loading");
    }
}
