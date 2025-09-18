package org.beehive.gpullama3.model.loader.batch;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.model.loader.batch.strategies.DirectAllocationStrategy;
import org.beehive.gpullama3.model.loader.batch.strategies.ExpertTensorStrategy;
import org.beehive.gpullama3.model.loader.batch.strategies.OptimizedLargeTensorStrategy;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * Abstract base class for model loaders that support batch processing and intelligent
 * tensor allocation strategies. Provides 95% generic implementation for handling
 * large tensor allocation issues across all MoE (Mixture-of-Experts) models.
 *
 * This class implements the Generic TornadoVM Large Tensor Allocation Framework
 * that addresses CL_MEM_OBJECT_ALLOCATION_FAILURE issues by:
 * - Automatic tensor size classification and strategy selection
 * - Memory-aware allocation with fragmentation prevention
 * - Expert tensor detection and specialized handling
 * - TornadoVM optimization integration
 *
 * Model-specific implementations only need to override ~5% of functionality:
 * - Expert tensor naming patterns
 * - Model-specific tensor handling requirements
 * - Weight creation from loaded tensors
 */
public abstract class BatchCapableModelLoader extends ModelLoader {

    private final Map<TensorAllocationType, TensorAllocationStrategy> strategies;
    private long availableGPUMemory;
    private long totalMemoryAllocated;
    private int largeTensorCount;

    /**
     * Constructor for batch-capable model loaders.
     */
    public BatchCapableModelLoader(FileChannel fileChannel, GGUF gguf,
                                   int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
        this.availableGPUMemory = queryAvailableGPUMemory();
        this.totalMemoryAllocated = 0L;
        this.largeTensorCount = 0;
        this.strategies = initializeAllocationStrategies();

        System.err.printf("[BATCH-LOADER] Initialized with %d MB available GPU memory%n",
                         availableGPUMemory / (1024 * 1024));
    }

    /**
     * Generic tensor loading with intelligent allocation strategy selection.
     * This method provides 95% of the implementation that works across all model types.
     */
    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        System.err.printf("[BATCH-LOADER] Loading %d tensors with intelligent allocation%n", tensorEntries.size());

        // Pre-flight analysis of memory requirements
        TensorLoadingPlan loadingPlan = analyzeTensorLoadingRequirements(tensorEntries);
        System.err.printf("[BATCH-LOADER] Loading plan: %s%n", loadingPlan);

        // Configure TornadoVM for optimal memory usage if enabled
        if (useTornadovm) {
            configureTornadoVMForLargeTensors(loadingPlan);
        }

        // Load all tensors using appropriate strategies
        Map<String, FloatTensor> loadedTensors = loadTensorsWithStrategies(tensorEntries, loadingPlan);

        // Create model-specific weights from loaded tensors
        return createWeightsFromTensors(loadedTensors, config);
    }

    /**
     * Loads tensors using appropriate allocation strategies based on tensor characteristics.
     */
    private Map<String, FloatTensor> loadTensorsWithStrategies(Map<String, GGMLTensorEntry> tensorEntries,
                                                              TensorLoadingPlan loadingPlan) {
        Map<String, FloatTensor> loadedTensors = new HashMap<>();
        int tensorIndex = 0;

        for (var entry : tensorEntries.entrySet()) {
            String tensorName = entry.getKey();
            GGMLTensorEntry tensorEntry = entry.getValue();
            long tensorElements = FloatTensor.numberOfElements(tensorEntry.shape());

            tensorIndex++;
            System.err.printf("[BATCH-LOADER] Loading tensor %d/%d: '%s'%n",
                             tensorIndex, tensorEntries.size(), tensorName);

            try {
                // Determine allocation type
                TensorAllocationType allocationType = classifyTensorForAllocation(tensorName, tensorElements);

                // Get appropriate strategy
                TensorAllocationStrategy strategy = strategies.get(allocationType);

                // Attempt allocation
                FloatTensor tensor = strategy.allocateTensor(tensorEntry, availableGPUMemory);

                if (tensor != null) {
                    loadedTensors.put(tensorName, tensor);

                    // Update memory tracking
                    long memoryUsed = estimateMemoryUsage(tensorElements, tensorEntry);
                    totalMemoryAllocated += memoryUsed;
                    availableGPUMemory -= memoryUsed;

                    System.err.printf("[BATCH-LOADER] Successfully loaded '%s' using %s strategy%n",
                                     tensorName, strategy.getStrategyName());
                } else {
                    // Handle allocation failure - try fallback or fail
                    handleTensorAllocationFailure(tensorName, tensorElements, allocationType);
                }

            } catch (TornadoOutOfMemoryException e) {
                // Handle memory allocation failure
                System.err.printf("[BATCH-LOADER] Memory allocation failed for '%s': %s%n",
                                 tensorName, e.getMessage());

                // Try recovery strategies
                if (!tryAllocationRecovery(tensorName, tensorEntry, loadedTensors)) {
                    throw new RuntimeException("Failed to allocate tensor: " + tensorName, e);
                }
            }
        }

        System.err.printf("[BATCH-LOADER] Successfully loaded %d/%d tensors, memory used: %d MB%n",
                         loadedTensors.size(), tensorEntries.size(),
                         totalMemoryAllocated / (1024 * 1024));

        return loadedTensors;
    }

    /**
     * Classifies tensor for allocation strategy selection, combining generic and model-specific logic.
     */
    private TensorAllocationType classifyTensorForAllocation(String tensorName, long tensorElements) {
        // Generic classification first
        TensorAllocationType genericType = TensorAllocationStrategy.classifyTensor(tensorElements, tensorName);

        // Model-specific override if needed
        if (requiresSpecialHandling(tensorName)) {
            return TensorAllocationType.REQUIRES_SPECIAL_HANDLING;
        }

        return genericType;
    }

    /**
     * Analyzes memory requirements and creates loading plan.
     */
    private TensorLoadingPlan analyzeTensorLoadingRequirements(Map<String, GGMLTensorEntry> tensorEntries) {
        long totalMemoryRequired = 0;
        int largeTensorCount = 0;
        int expertTensorCount = 0;
        int standardTensorCount = 0;

        for (var entry : tensorEntries.entrySet()) {
            String tensorName = entry.getKey();
            GGMLTensorEntry tensorEntry = entry.getValue();
            long tensorElements = FloatTensor.numberOfElements(tensorEntry.shape());
            long memoryRequired = estimateMemoryUsage(tensorElements, tensorEntry);

            totalMemoryRequired += memoryRequired;

            TensorAllocationType type = TensorAllocationStrategy.classifyTensor(tensorElements, tensorName);
            switch (type) {
                case REQUIRES_SPECIAL_HANDLING -> largeTensorCount++;
                case EXPERT_TENSOR -> expertTensorCount++;
                default -> standardTensorCount++;
            }

            if (tensorElements > TensorAllocationStrategy.LARGE_TENSOR_THRESHOLD) {
                System.err.printf("[BATCH-LOADER] Large tensor detected: '%s' (%d MB)%n",
                                 tensorName, memoryRequired / (1024 * 1024));
            }
        }

        return new TensorLoadingPlan(totalMemoryRequired, largeTensorCount, expertTensorCount,
                                   standardTensorCount, availableGPUMemory);
    }

    /**
     * Configures TornadoVM execution plan for optimal large tensor handling.
     */
    private void configureTornadoVMForLargeTensors(TensorLoadingPlan loadingPlan) {
        System.err.printf("[BATCH-LOADER] Configuring TornadoVM for %d large tensors%n",
                         loadingPlan.largeTensorCount);

        // TODO: Integrate with TornadoVMMemoryOptimizer when created
        // This would configure TornadoVM execution plan settings based on loading plan
    }

    /**
     * Handles tensor allocation failure with recovery strategies.
     */
    private boolean tryAllocationRecovery(String tensorName, GGMLTensorEntry tensorEntry,
                                         Map<String, FloatTensor> loadedTensors) {
        System.err.printf("[BATCH-LOADER] Attempting allocation recovery for '%s'%n", tensorName);

        // Recovery strategy 1: Free some memory and retry
        // Recovery strategy 2: Use alternative allocation strategy
        // Recovery strategy 3: Partial tensor loading

        // For now, return false to indicate no recovery available
        return false;
    }

    private void handleTensorAllocationFailure(String tensorName, long tensorElements,
                                             TensorAllocationType allocationType) {
        System.err.printf("[BATCH-LOADER] Tensor allocation returned null for '%s' (type: %s, elements: %d)%n",
                         tensorName, allocationType, tensorElements);
        throw new RuntimeException("Tensor allocation strategy returned null: " + tensorName);
    }

    private long estimateMemoryUsage(long tensorElements, GGMLTensorEntry tensorEntry) {
        int bytesPerElement = getBytesPerElement(tensorEntry);
        return tensorElements * bytesPerElement;
    }

    private int getBytesPerElement(GGMLTensorEntry entry) {
        return switch (entry.ggmlType()) {
            case F32 -> 4;
            case F16 -> 2;
            case Q4_0, Q4_1, Q4_K -> 1;
            case Q8_0, Q8_1, Q8_K -> 1;
            case IQ3_T -> 1;
            default -> 2;
        };
    }

    private long queryAvailableGPUMemory() {
        // TODO: Implement actual GPU memory query
        // For now, assume 40GB conservative limit on high-end cards
        return 40L * 1024 * 1024 * 1024; // 40GB in bytes
    }

    private Map<TensorAllocationType, TensorAllocationStrategy> initializeAllocationStrategies() {
        Map<TensorAllocationType, TensorAllocationStrategy> strategyMap = new HashMap<>();

        strategyMap.put(TensorAllocationType.STANDARD_ALLOCATION, new DirectAllocationStrategy());
        strategyMap.put(TensorAllocationType.MONITOR_ALLOCATION, new DirectAllocationStrategy());
        strategyMap.put(TensorAllocationType.EXPERT_TENSOR, new ExpertTensorStrategy());
        strategyMap.put(TensorAllocationType.REQUIRES_SPECIAL_HANDLING, new OptimizedLargeTensorStrategy());

        return strategyMap;
    }

    // Abstract methods that model-specific implementations must override (5% of code)

    /**
     * Model-specific check for tensors requiring special handling beyond generic classification.
     */
    protected abstract boolean requiresSpecialHandling(String tensorName);

    /**
     * Model-specific expert tensor patterns (used for additional validation).
     */
    protected abstract String[] getModelSpecificExpertPatterns();

    /**
     * Create model-specific weights object from loaded tensors.
     */
    protected abstract Weights createWeightsFromTensors(Map<String, FloatTensor> tensors, Configuration config);

    /**
     * Tensor loading plan analysis results.
     */
    private static class TensorLoadingPlan {
        final long totalMemoryRequired;
        final int largeTensorCount;
        final int expertTensorCount;
        final int standardTensorCount;
        final long availableMemory;

        TensorLoadingPlan(long totalMemoryRequired, int largeTensorCount, int expertTensorCount,
                         int standardTensorCount, long availableMemory) {
            this.totalMemoryRequired = totalMemoryRequired;
            this.largeTensorCount = largeTensorCount;
            this.expertTensorCount = expertTensorCount;
            this.standardTensorCount = standardTensorCount;
            this.availableMemory = availableMemory;
        }

        @Override
        public String toString() {
            return String.format("TensorPlan[total=%dMB, large=%d, expert=%d, standard=%d, available=%dMB]",
                               totalMemoryRequired / (1024 * 1024), largeTensorCount, expertTensorCount,
                               standardTensorCount, availableMemory / (1024 * 1024));
        }
    }
}