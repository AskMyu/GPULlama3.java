package org.beehive.gpullama3.model.loader.batch;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;

import java.util.Arrays;

/**
 * Generic tensor allocation strategy interface for handling different tensor sizes
 * and characteristics in TornadoVM environment. This interface provides the foundation
 * for memory-aware tensor allocation across all model types.
 */
public interface TensorAllocationStrategy {

    // Memory thresholds for tensor classification (in elements)
    long LARGE_TENSOR_THRESHOLD = 200_000_000L;  // 200M elements ≈ 400MB
    long EXPERT_TENSOR_THRESHOLD = 100_000_000L;  // 100M elements ≈ 200MB
    long FRAGMENTATION_RISK_SIZE = 50_000_000L;   // 50M elements ≈ 100MB

    /**
     * Allocates a tensor using the specific strategy implementation.
     *
     * @param entry The GGML tensor entry containing tensor metadata and memory segment
     * @param availableMemory Current available GPU memory in bytes
     * @return Allocated FloatTensor ready for use
     * @throws TornadoOutOfMemoryException if allocation fails
     */
    FloatTensor allocateTensor(GGMLTensorEntry entry, long availableMemory) throws TornadoOutOfMemoryException;

    /**
     * Checks if the strategy can allocate a tensor of the given size with available memory.
     *
     * @param tensorSizeElements Number of elements in the tensor
     * @param availableMemory Available GPU memory in bytes
     * @return true if allocation is likely to succeed, false otherwise
     */
    boolean canAllocate(long tensorSizeElements, long availableMemory);

    /**
     * Returns a human-readable name for this allocation strategy for logging.
     *
     * @return Strategy name
     */
    String getStrategyName();

    /**
     * Generic tensor classification based on size and name patterns.
     * This method provides universal tensor categorization across all model types.
     *
     * @param tensorElements Number of elements in the tensor
     * @param tensorName Name of the tensor (used for expert detection)
     * @return Appropriate allocation type for the tensor
     */
    static TensorAllocationType classifyTensor(long tensorElements, String tensorName) {
        if (tensorElements > LARGE_TENSOR_THRESHOLD) {
            return TensorAllocationType.REQUIRES_SPECIAL_HANDLING;
        } else if (tensorElements > EXPERT_TENSOR_THRESHOLD && isExpertTensorByName(tensorName)) {
            return TensorAllocationType.EXPERT_TENSOR;
        } else if (tensorElements > FRAGMENTATION_RISK_SIZE) {
            return TensorAllocationType.MONITOR_ALLOCATION;
        } else {
            return TensorAllocationType.STANDARD_ALLOCATION;
        }
    }

    /**
     * Generic expert tensor pattern matching across different MoE architectures.
     * Detects expert tensors based on common naming conventions used by various models.
     *
     * @param tensorName The tensor name to analyze
     * @return true if tensor appears to be an expert tensor, false otherwise
     */
    static boolean isExpertTensorByName(String tensorName) {
        if (tensorName == null) {
            return false;
        }

        String lowerName = tensorName.toLowerCase();
        String[] commonExpertPatterns = {
            "_exps.",     // GPT-OSS, OLMoE: ffn_gate_exps, ffn_down_exps, ffn_up_exps
            "expert",     // General expert naming
            "moe",        // Mixture-of-experts indicator
            "ffn_gate",   // Feed-forward network gate projections
            "ffn_up",     // Feed-forward network up projections
            "ffn_down",   // Feed-forward network down projections
            "gate_proj",  // QwenMoE, DeepSeekMoE patterns
            "up_proj",    // QwenMoE, DeepSeekMoE patterns
            "down_proj"   // QwenMoE, DeepSeekMoE patterns
        };

        return Arrays.stream(commonExpertPatterns)
                    .anyMatch(lowerName::contains);
    }

    /**
     * Estimates memory requirements for a tensor in bytes based on element count and type.
     *
     * @param tensorElements Number of elements in the tensor
     * @param bytesPerElement Bytes per element (2 for Float16, 4 for Float32, etc.)
     * @return Estimated memory requirement in bytes
     */
    static long estimateMemoryBytes(long tensorElements, int bytesPerElement) {
        return tensorElements * bytesPerElement;
    }

    /**
     * Estimates memory requirements in megabytes for logging purposes.
     *
     * @param tensorElements Number of elements in the tensor
     * @param bytesPerElement Bytes per element
     * @return Estimated memory requirement in MB
     */
    static long estimateMemoryMB(long tensorElements, int bytesPerElement) {
        return estimateMemoryBytes(tensorElements, bytesPerElement) / (1024 * 1024);
    }
}