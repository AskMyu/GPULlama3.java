package org.beehive.gpullama3.model.loader.batch.strategies;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.model.loader.batch.TensorAllocationStrategy;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

/**
 * Expert-specific allocation strategy for MoE (Mixture-of-Experts) model tensors.
 * Handles expert network weights with specialized memory management optimized
 * for the expert tensor access patterns and memory requirements.
 *
 * Expert tensors typically have characteristics:
 * - Large size (100M-300M elements each)
 * - Sequential access patterns during inference
 * - Multiple experts per layer (8-64 experts common)
 * - Often quantized (IQ3_T, Q4_K, Q8_K)
 */
public class ExpertTensorStrategy implements TensorAllocationStrategy {

    private static final double EXPERT_SAFETY_MARGIN = 0.75; // Conservative margin for expert tensors
    private static final long EXPERT_TENSOR_MIN_SIZE = 50_000_000L; // 50M elements minimum for expert classification

    @Override
    public FloatTensor allocateTensor(GGMLTensorEntry entry, long availableMemory) throws TornadoOutOfMemoryException {
        long tensorElements = FloatTensor.numberOfElements(entry.shape());
        int bytesPerElement = getBytesPerElement(entry);
        long memoryRequired = TensorAllocationStrategy.estimateMemoryBytes(tensorElements, bytesPerElement);

        System.err.printf("[EXPERT-ALLOC] Loading expert tensor '%s' (%d MB, %d elements)%n",
                         entry.name(),
                         memoryRequired / (1024 * 1024),
                         tensorElements);

        if (!canAllocate(tensorElements, availableMemory)) {
            throw new TornadoOutOfMemoryException(String.format(
                "Cannot allocate expert tensor '%s': requires %d MB, available %d MB",
                entry.name(),
                memoryRequired / (1024 * 1024),
                (long) (availableMemory * EXPERT_SAFETY_MARGIN / (1024 * 1024))
            ));
        }

        // Analyze expert tensor characteristics
        ExpertTensorInfo expertInfo = analyzeExpertTensor(entry);
        System.err.printf("[EXPERT-ALLOC] Expert analysis: %s%n", expertInfo);

        // Try expert-specific allocation strategies
        FloatTensor tensor = tryExpertOptimizedAllocation(entry, expertInfo);
        if (tensor != null) {
            System.err.printf("[EXPERT-ALLOC] Successfully allocated expert tensor '%s'%n", entry.name());
            return tensor;
        }

        // Fallback to optimized large tensor strategy
        System.err.printf("[EXPERT-ALLOC] Using large tensor fallback for '%s'%n", entry.name());
        return tryLargeTensorFallback(entry);
    }

    @Override
    public boolean canAllocate(long tensorSizeElements, long availableMemory) {
        // Expert tensors need more conservative memory management
        long estimatedMemory = TensorAllocationStrategy.estimateMemoryBytes(tensorSizeElements, 2);
        return estimatedMemory < (availableMemory * EXPERT_SAFETY_MARGIN);
    }

    @Override
    public String getStrategyName() {
        return "ExpertTensor";
    }

    /**
     * Analyzes expert tensor characteristics to optimize allocation strategy.
     */
    private ExpertTensorInfo analyzeExpertTensor(GGMLTensorEntry entry) {
        String tensorName = entry.name();
        long tensorElements = FloatTensor.numberOfElements(entry.shape());

        // Determine expert tensor type
        ExpertTensorType type = identifyExpertTensorType(tensorName);

        // Estimate layer and expert indices if possible
        int layerIndex = extractLayerIndex(tensorName);
        int expertIndex = extractExpertIndex(tensorName);

        return new ExpertTensorInfo(type, layerIndex, expertIndex, tensorElements, entry.ggmlType());
    }

    /**
     * Identifies the type of expert tensor based on naming patterns.
     */
    private ExpertTensorType identifyExpertTensorType(String tensorName) {
        String lowerName = tensorName.toLowerCase();

        if (lowerName.contains("gate") || lowerName.contains("router")) {
            return ExpertTensorType.GATE_ROUTER;
        } else if (lowerName.contains("ffn_gate") || lowerName.contains("gate_proj")) {
            return ExpertTensorType.GATE_PROJECTION;
        } else if (lowerName.contains("ffn_up") || lowerName.contains("up_proj")) {
            return ExpertTensorType.UP_PROJECTION;
        } else if (lowerName.contains("ffn_down") || lowerName.contains("down_proj")) {
            return ExpertTensorType.DOWN_PROJECTION;
        } else {
            return ExpertTensorType.UNKNOWN_EXPERT;
        }
    }

    /**
     * Extracts layer index from tensor name (e.g., "blk.5.ffn_gate_exps" -> 5).
     */
    private int extractLayerIndex(String tensorName) {
        try {
            // Look for "blk.N." pattern
            if (tensorName.contains("blk.")) {
                int blkStart = tensorName.indexOf("blk.") + 4;
                int blkEnd = tensorName.indexOf(".", blkStart);
                if (blkEnd > blkStart) {
                    return Integer.parseInt(tensorName.substring(blkStart, blkEnd));
                }
            }
            // Look for "layers.N." pattern
            if (tensorName.contains("layers.")) {
                int layersStart = tensorName.indexOf("layers.") + 7;
                int layersEnd = tensorName.indexOf(".", layersStart);
                if (layersEnd > layersStart) {
                    return Integer.parseInt(tensorName.substring(layersStart, layersEnd));
                }
            }
        } catch (NumberFormatException e) {
            // Ignore parsing errors
        }
        return -1; // Unknown layer
    }

    /**
     * Extracts expert index if present in tensor name.
     */
    private int extractExpertIndex(String tensorName) {
        // Most expert tensors don't have individual expert indices in GGUF format
        // They're typically stored as combined tensors with all experts
        return -1; // Combined expert tensor
    }

    /**
     * Tries expert-specific optimized allocation strategies.
     */
    private FloatTensor tryExpertOptimizedAllocation(GGMLTensorEntry entry, ExpertTensorInfo expertInfo) {
        try {
            // Strategy 1: Expert tensor sequential allocation
            FloatTensor tensor = trySequentialExpertAllocation(entry, expertInfo);
            if (tensor != null) {
                return tensor;
            }

            // Strategy 2: Expert-specific memory alignment
            tensor = tryExpertAlignedAllocation(entry, expertInfo);
            if (tensor != null) {
                return tensor;
            }

            // Strategy 3: Quantization-aware allocation
            tensor = tryQuantizationAwareAllocation(entry, expertInfo);
            return tensor;

        } catch (Exception e) {
            System.err.printf("[EXPERT-ALLOC] Expert optimization failed for '%s': %s%n",
                             entry.name(), e.getMessage());
            return null;
        }
    }

    private FloatTensor trySequentialExpertAllocation(GGMLTensorEntry entry, ExpertTensorInfo expertInfo) {
        System.err.printf("[EXPERT-ALLOC] Trying sequential allocation for %s expert '%s'%n",
                         expertInfo.type, entry.name());
        // TODO: Implement sequential allocation optimized for expert access patterns
        return null;
    }

    private FloatTensor tryExpertAlignedAllocation(GGMLTensorEntry entry, ExpertTensorInfo expertInfo) {
        System.err.printf("[EXPERT-ALLOC] Trying aligned allocation for %s expert '%s'%n",
                         expertInfo.type, entry.name());
        // TODO: Implement expert-specific memory alignment
        return null;
    }

    private FloatTensor tryQuantizationAwareAllocation(GGMLTensorEntry entry, ExpertTensorInfo expertInfo) {
        System.err.printf("[EXPERT-ALLOC] Trying quantization-aware allocation for %s expert '%s' (type: %s)%n",
                         expertInfo.type, entry.name(), expertInfo.ggmlType);
        // TODO: Implement allocation optimized for specific quantization types (IQ3_T, Q4_K, etc.)
        return null;
    }

    private FloatTensor tryLargeTensorFallback(GGMLTensorEntry entry) throws TornadoOutOfMemoryException {
        System.err.printf("[EXPERT-ALLOC] Using large tensor strategy fallback for '%s'%n", entry.name());

        try {
            // Use ModelLoader's standard quantized tensor loading as ultimate fallback
            return org.beehive.gpullama3.model.loader.ModelLoader.loadQuantized(entry);
        } catch (Exception e) {
            System.err.printf("[EXPERT-ALLOC-ERROR] Large tensor fallback failed for '%s': %s%n",
                             entry.name(), e.getMessage());
            throw new TornadoOutOfMemoryException("Expert tensor allocation fallback failed: " + e.getMessage());
        }
    }

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

    /**
     * Expert tensor type classification.
     */
    private enum ExpertTensorType {
        GATE_ROUTER,        // Router/gating weights
        GATE_PROJECTION,    // Gate projection in expert FFN
        UP_PROJECTION,      // Up projection in expert FFN
        DOWN_PROJECTION,    // Down projection in expert FFN
        UNKNOWN_EXPERT      // Expert tensor of unknown type
    }

    /**
     * Expert tensor analysis information.
     */
    private static class ExpertTensorInfo {
        final ExpertTensorType type;
        final int layerIndex;
        final int expertIndex;
        final long tensorElements;
        final org.beehive.gpullama3.core.model.GGMLType ggmlType;

        ExpertTensorInfo(ExpertTensorType type, int layerIndex, int expertIndex,
                        long tensorElements, org.beehive.gpullama3.core.model.GGMLType ggmlType) {
            this.type = type;
            this.layerIndex = layerIndex;
            this.expertIndex = expertIndex;
            this.tensorElements = tensorElements;
            this.ggmlType = ggmlType;
        }

        @Override
        public String toString() {
            return String.format("ExpertTensor[type=%s, layer=%d, expert=%d, elements=%d, quantization=%s]",
                               type, layerIndex, expertIndex, tensorElements, ggmlType);
        }
    }
}