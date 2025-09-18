package org.beehive.gpullama3.model.loader.batch;

/**
 * Enum defining different types of tensor allocation strategies based on tensor size
 * and characteristics. Used by the Generic TornadoVM Large Tensor Allocation Framework
 * to determine appropriate memory allocation strategies.
 */
public enum TensorAllocationType {

    /**
     * Standard tensor allocation for small to medium tensors.
     * Uses direct memory allocation without special handling.
     * Suitable for tensors < 50M elements (~100MB).
     */
    STANDARD_ALLOCATION,

    /**
     * Monitor allocation for medium-sized tensors that may cause fragmentation.
     * Uses standard allocation but with memory usage monitoring.
     * Suitable for tensors 50M - 100M elements (100MB - 200MB).
     */
    MONITOR_ALLOCATION,

    /**
     * Expert tensor allocation for MoE (Mixture-of-Experts) model tensors.
     * Uses specialized handling for expert network weights.
     * Suitable for expert tensors 100M - 200M elements (200MB - 400MB).
     */
    EXPERT_TENSOR,

    /**
     * Special handling for very large tensors that require memory optimization.
     * Uses advanced allocation strategies to prevent fragmentation failures.
     * Suitable for tensors > 200M elements (> 400MB).
     */
    REQUIRES_SPECIAL_HANDLING
}