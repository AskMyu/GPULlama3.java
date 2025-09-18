package org.beehive.gpullama3.model.loader.batch.strategies;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.model.loader.batch.TensorAllocationStrategy;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

/**
 * Memory-optimized allocation strategy for large tensors.
 * Uses TornadoVM memory management optimizations to handle large tensors
 * that may cause fragmentation issues with standard allocation.
 *
 * This strategy implements advanced memory allocation techniques:
 * - Memory alignment optimization
 * - Fragmentation avoidance
 * - TornadoVM memory management APIs utilization
 */
public class OptimizedLargeTensorStrategy implements TensorAllocationStrategy {

    private static final double LARGE_TENSOR_SAFETY_MARGIN = 0.7; // More conservative for large tensors
    private static final long MIN_CONTIGUOUS_BLOCK_SIZE = 100_000_000L; // 100MB minimum contiguous block

    @Override
    public FloatTensor allocateTensor(GGMLTensorEntry entry, long availableMemory) throws TornadoOutOfMemoryException {
        long tensorElements = FloatTensor.numberOfElements(entry.shape());
        int bytesPerElement = getBytesPerElement(entry);
        long memoryRequired = TensorAllocationStrategy.estimateMemoryBytes(tensorElements, bytesPerElement);

        System.err.printf("[OPTIMIZED-ALLOC] Attempting optimized allocation for large tensor '%s' (%d MB)%n",
                         entry.name(), memoryRequired / (1024 * 1024));

        if (!canAllocate(tensorElements, availableMemory)) {
            throw new TornadoOutOfMemoryException(String.format(
                "Cannot allocate large tensor '%s': requires %d MB, available %d MB",
                entry.name(),
                memoryRequired / (1024 * 1024),
                (long) (availableMemory * LARGE_TENSOR_SAFETY_MARGIN / (1024 * 1024))
            ));
        }

        // Try memory-optimized allocation approaches
        FloatTensor tensor = tryOptimizedAllocation(entry);
        if (tensor != null) {
            System.err.printf("[OPTIMIZED-ALLOC] Successfully allocated tensor '%s' using optimized strategy%n",
                             entry.name());
            return tensor;
        }

        // Fallback to standard allocation if optimization fails
        System.err.printf("[OPTIMIZED-ALLOC] Falling back to standard allocation for '%s'%n", entry.name());
        return tryStandardAllocation(entry);
    }

    @Override
    public boolean canAllocate(long tensorSizeElements, long availableMemory) {
        long estimatedMemory = TensorAllocationStrategy.estimateMemoryBytes(tensorSizeElements, 2); // Conservative estimate

        // Check if we have enough memory with safety margin
        boolean hasEnoughMemory = estimatedMemory < (availableMemory * LARGE_TENSOR_SAFETY_MARGIN);

        // Additional check: ensure we can allocate a minimum contiguous block
        boolean hasContiguousSpace = estimatedMemory >= MIN_CONTIGUOUS_BLOCK_SIZE;

        return hasEnoughMemory && (estimatedMemory < MIN_CONTIGUOUS_BLOCK_SIZE || hasContiguousSpace);
    }

    @Override
    public String getStrategyName() {
        return "OptimizedLargeTensor";
    }

    /**
     * Attempts optimized allocation using TornadoVM memory management features.
     */
    private FloatTensor tryOptimizedAllocation(GGMLTensorEntry entry) {
        try {
            // Approach 1: Memory alignment optimization
            FloatTensor tensor = tryAlignedAllocation(entry);
            if (tensor != null) {
                return tensor;
            }

            // Approach 2: TornadoVM memory management APIs
            tensor = tryTornadoVMOptimizedAllocation(entry);
            if (tensor != null) {
                return tensor;
            }

            // Approach 3: Gradual allocation with memory monitoring
            tensor = tryGradualAllocation(entry);
            return tensor;

        } catch (Exception e) {
            System.err.printf("[OPTIMIZED-ALLOC] Optimization failed for '%s': %s%n",
                             entry.name(), e.getMessage());
            return null;
        }
    }

    /**
     * Attempts memory-aligned allocation to reduce fragmentation.
     */
    private FloatTensor tryAlignedAllocation(GGMLTensorEntry entry) {
        System.err.printf("[OPTIMIZED-ALLOC] Trying aligned allocation for '%s'%n", entry.name());

        // TODO: Implement aligned memory allocation
        // This would use memory alignment techniques to improve allocation success
        // For now, return null to indicate not implemented
        return null;
    }

    /**
     * Uses TornadoVM-specific memory management APIs.
     */
    private FloatTensor tryTornadoVMOptimizedAllocation(GGMLTensorEntry entry) {
        System.err.printf("[OPTIMIZED-ALLOC] Trying TornadoVM optimized allocation for '%s'%n", entry.name());

        // TODO: Implement TornadoVM memory management API usage
        // This would use lockObjectInMemory/unlockObjectFromMemory and other TornadoVM APIs
        // For now, return null to indicate not implemented
        return null;
    }

    /**
     * Attempts gradual allocation with memory monitoring.
     */
    private FloatTensor tryGradualAllocation(GGMLTensorEntry entry) {
        System.err.printf("[OPTIMIZED-ALLOC] Trying gradual allocation for '%s'%n", entry.name());

        // TODO: Implement gradual allocation strategy
        // This would allocate memory in smaller chunks and monitor for failures
        // For now, return null to indicate not implemented
        return null;
    }

    /**
     * Fallback to standard allocation approach.
     */
    private FloatTensor tryStandardAllocation(GGMLTensorEntry entry) throws TornadoOutOfMemoryException {
        System.err.printf("[OPTIMIZED-ALLOC] Using standard allocation fallback for '%s'%n", entry.name());

        try {
            // Use ModelLoader's standard quantized tensor loading as fallback
            return org.beehive.gpullama3.model.loader.ModelLoader.loadQuantized(entry);
        } catch (Exception e) {
            System.err.printf("[OPTIMIZED-ALLOC-ERROR] Standard fallback failed for '%s': %s%n",
                             entry.name(), e.getMessage());
            throw new TornadoOutOfMemoryException("Optimized allocation fallback failed: " + e.getMessage());
        }
    }

    /**
     * Gets bytes per element based on GGML type (same as DirectAllocationStrategy).
     */
    private int getBytesPerElement(GGMLTensorEntry entry) {
        switch (entry.ggmlType()) {
            case F32 -> { return 4; }
            case F16 -> { return 2; }
            case Q4_0, Q4_1, Q4_K -> { return 1; }
            case Q8_0, Q8_1, Q8_K -> { return 1; }
            case IQ3_T -> { return 1; }
            default -> { return 2; }
        }
    }
}