package org.beehive.gpullama3.model.loader.batch.strategies;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.model.loader.batch.TensorAllocationStrategy;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

/**
 * Direct allocation strategy for standard tensor loading.
 * Uses the existing ModelLoader tensor allocation approach without modifications.
 * Suitable for small to medium tensors that don't cause memory fragmentation issues.
 */
public class DirectAllocationStrategy implements TensorAllocationStrategy {

    private static final double SAFETY_MARGIN = 0.8; // Use 80% of available memory for safety

    @Override
    public FloatTensor allocateTensor(GGMLTensorEntry entry, long availableMemory) throws TornadoOutOfMemoryException {
        long tensorElements = FloatTensor.numberOfElements(entry.shape());
        int bytesPerElement = getBytesPerElement(entry);

        if (!canAllocate(tensorElements, availableMemory)) {
            throw new TornadoOutOfMemoryException(String.format(
                "Cannot allocate tensor '%s': requires %d MB, available %d MB (with safety margin)",
                entry.name(),
                TensorAllocationStrategy.estimateMemoryMB(tensorElements, bytesPerElement),
                (long) (availableMemory * SAFETY_MARGIN / (1024 * 1024))
            ));
        }

        System.err.printf("[DIRECT-ALLOC] Loading tensor '%s' (%d MB, %d elements)%n",
                         entry.name(),
                         TensorAllocationStrategy.estimateMemoryMB(tensorElements, bytesPerElement),
                         tensorElements);

        // Use standard tensor loading from ModelLoader
        return loadStandardTensor(entry);
    }

    @Override
    public boolean canAllocate(long tensorSizeElements, long availableMemory) {
        // Estimate memory requirement based on typical quantization (assume 2 bytes average)
        long estimatedMemory = tensorSizeElements * 2L; // Conservative estimate
        return estimatedMemory < (availableMemory * SAFETY_MARGIN);
    }

    @Override
    public String getStrategyName() {
        return "DirectAllocation";
    }

    /**
     * Loads tensor using standard ModelLoader approach.
     * Uses ModelLoader.loadQuantized() which handles all quantization types correctly.
     */
    private FloatTensor loadStandardTensor(GGMLTensorEntry entry) {
        try {
            // Use the static method from ModelLoader for standard tensor loading
            return org.beehive.gpullama3.model.loader.ModelLoader.loadQuantized(entry);
        } catch (Exception e) {
            System.err.printf("[DIRECT-ALLOC-ERROR] Failed to load tensor '%s': %s%n",
                             entry.name(), e.getMessage());
            throw new uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException(
                "Direct allocation failed: " + e.getMessage());
        }
    }

    /**
     * Gets bytes per element based on GGML type.
     */
    private int getBytesPerElement(GGMLTensorEntry entry) {
        // Use GGML type to determine bytes per element
        switch (entry.ggmlType()) {
            case F32 -> { return 4; }
            case F16 -> { return 2; }
            case Q4_0, Q4_1, Q4_K -> { return 1; } // Approximation for quantized
            case Q8_0, Q8_1, Q8_K -> { return 1; }
            case IQ3_T -> { return 1; } // Approximation for ternary quantization
            default -> { return 2; } // Conservative default
        }
    }
}