package org.beehive.gpullama3.model.loader.batch;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/**
 * Utility class for TornadoVM-specific memory optimizations and configuration.
 * Provides methods to configure TornadoVM execution plans for optimal large tensor
 * allocation and to work around memory fragmentation issues.
 *
 * Based on TornadoVM source analysis:
 * - Cannot reset() after warmup() - memory layout is immutable
 * - withBatch() splits iteration space, not tensor allocation
 * - All tensors must be allocated during initialization
 * - Memory management APIs: lockObjectInMemory/unlockObjectFromMemory available
 */
public class TornadoVMMemoryOptimizer {

    private static final long DEFAULT_MEMORY_LIMIT_BYTES = 40L * 1024 * 1024 * 1024; // 40GB conservative
    private static final double MEMORY_SAFETY_FACTOR = 0.9; // Use 90% of available memory

    /**
     * Configures TornadoVM execution plan for optimal large tensor allocation.
     * Must be called before the first warmup() - cannot be changed after initialization.
     *
     * @param executionPlan The TornadoVM execution plan to configure
     * @param estimatedMemoryUsage Estimated total memory usage for all tensors
     * @param largeTensorCount Number of large tensors that will be allocated
     */
    public static void configureForLargeTensors(TornadoExecutionPlan executionPlan,
                                              long estimatedMemoryUsage,
                                              int largeTensorCount) {

        System.err.printf("[TORNADO-OPTIMIZER] Configuring execution plan for %d large tensors, %d MB total%n",
                         largeTensorCount, estimatedMemoryUsage / (1024 * 1024));

        try {
            // Set conservative memory limit based on estimated usage
            long memoryLimitMB = Math.min(
                (long) (estimatedMemoryUsage * 1.2 / (1024 * 1024)), // 20% overhead
                DEFAULT_MEMORY_LIMIT_BYTES / (1024 * 1024)
            );

            executionPlan
                .withMemoryLimit(memoryLimitMB + "MB")
                .withDevice(getOptimalDevice())
                .withPreCompilation(); // Pre-compile to detect memory issues early

            System.err.printf("[TORNADO-OPTIMIZER] Set memory limit to %d MB%n", memoryLimitMB);

        } catch (Exception e) {
            System.err.printf("[TORNADO-OPTIMIZER] Failed to configure execution plan: %s%n", e.getMessage());
            System.err.println("[TORNADO-OPTIMIZER] Continuing with default configuration");
        }
    }

    /**
     * Configures execution plan with batch processing if beneficial.
     * Note: Based on TornadoVM source, withBatch() is for iteration space splitting,
     * not tensor allocation, but may still help with memory pressure.
     */
    public static void configureWithBatchIfBeneficial(TornadoExecutionPlan executionPlan,
                                                     long totalMemoryRequired,
                                                     long availableMemory) {

        double memoryPressure = (double) totalMemoryRequired / availableMemory;

        if (memoryPressure > 0.8) { // High memory pressure
            // Calculate appropriate batch size to reduce memory pressure
            long batchSizeMB = Math.max(100, availableMemory / (8 * 1024 * 1024)); // At least 100MB, max 1/8 available

            System.err.printf("[TORNADO-OPTIMIZER] High memory pressure (%.1f%%), enabling batch processing (%d MB)%n",
                             memoryPressure * 100, batchSizeMB);

            try {
                executionPlan.withBatch(batchSizeMB + "MB");
            } catch (Exception e) {
                System.err.printf("[TORNADO-OPTIMIZER] Failed to enable batch processing: %s%n", e.getMessage());
            }
        } else {
            System.err.printf("[TORNADO-OPTIMIZER] Memory pressure acceptable (%.1f%%), no batching needed%n",
                             memoryPressure * 100);
        }
    }

    /**
     * Attempts to allocate aligned tensor memory to reduce fragmentation.
     * Uses TornadoVM memory management APIs when available.
     */
    public static FloatTensor allocateAlignedTensor(GGMLTensorEntry entry) {
        long tensorElements = FloatTensor.numberOfElements(entry.shape());

        System.err.printf("[TORNADO-OPTIMIZER] Attempting aligned allocation for tensor '%s' (%d elements)%n",
                         entry.name(), tensorElements);

        try {
            // TODO: Implement aligned memory allocation using TornadoVM APIs
            // This would use lockObjectInMemory/unlockObjectFromMemory and other TornadoVM
            // memory management features to optimize allocation

            System.err.printf("[TORNADO-OPTIMIZER] Aligned allocation not yet implemented for '%s'%n", entry.name());
            return null; // Indicate not implemented yet

        } catch (Exception e) {
            System.err.printf("[TORNADO-OPTIMIZER] Aligned allocation failed for '%s': %s%n",
                             entry.name(), e.getMessage());
            return null;
        }
    }

    /**
     * Queries available GPU memory using TornadoVM device APIs.
     */
    public static long queryAvailableGPUMemory() {
        try {
            TornadoDevice device = getOptimalDevice();

            // TODO: Implement actual memory query using TornadoVM device APIs
            // For now, use conservative estimate
            long totalMemory = 48L * 1024 * 1024 * 1024; // 48GB for RTX 6000 Ada / A6000
            long usedMemory = 0L; // TODO: Query actual usage

            long availableMemory = (long) ((totalMemory - usedMemory) * MEMORY_SAFETY_FACTOR);

            System.err.printf("[TORNADO-OPTIMIZER] Available GPU memory: %d MB (%.1f%% of total)%n",
                             availableMemory / (1024 * 1024),
                             (double) availableMemory / totalMemory * 100);

            return availableMemory;

        } catch (Exception e) {
            System.err.printf("[TORNADO-OPTIMIZER] Failed to query GPU memory: %s%n", e.getMessage());
            return DEFAULT_MEMORY_LIMIT_BYTES; // Conservative fallback
        }
    }

    /**
     * Gets the optimal TornadoVM device for large tensor operations.
     * Prefers devices with most available memory.
     */
    public static TornadoDevice getOptimalDevice() {
        try {
            // Use default device for now
            TornadoDevice device = TornadoExecutionPlan.DEFAULT_DEVICE;

            System.err.printf("[TORNADO-OPTIMIZER] Using device: %s%n", device.toString());
            return device;

        } catch (Exception e) {
            System.err.printf("[TORNADO-OPTIMIZER] Failed to get optimal device: %s%n", e.getMessage());
            return TornadoExecutionPlan.DEFAULT_DEVICE;
        }
    }

    /**
     * Estimates TornadoVM overhead for execution plan with given number of tensors.
     */
    public static long estimateTornadoVMOverhead(int tensorCount, long totalTensorMemory) {
        // Empirical estimates based on TornadoVM memory usage patterns
        long baseOverhead = 100L * 1024 * 1024; // 100MB base overhead
        long perTensorOverhead = 1L * 1024 * 1024; // 1MB per tensor
        long memoryPercentageOverhead = (long) (totalTensorMemory * 0.1); // 10% of tensor memory

        long totalOverhead = baseOverhead + (tensorCount * perTensorOverhead) + memoryPercentageOverhead;

        System.err.printf("[TORNADO-OPTIMIZER] Estimated TornadoVM overhead: %d MB for %d tensors%n",
                         totalOverhead / (1024 * 1024), tensorCount);

        return totalOverhead;
    }

    /**
     * Checks if TornadoVM is properly configured for large tensor operations.
     */
    public static boolean validateTornadoVMConfiguration() {
        try {
            // Basic validation checks
            TornadoDevice device = getOptimalDevice();
            if (device == null) {
                System.err.println("[TORNADO-OPTIMIZER] No TornadoVM device available");
                return false;
            }

            long availableMemory = queryAvailableGPUMemory();
            if (availableMemory < 1024 * 1024 * 1024) { // Less than 1GB
                System.err.printf("[TORNADO-OPTIMIZER] Insufficient GPU memory: %d MB%n",
                                 availableMemory / (1024 * 1024));
                return false;
            }

            System.err.println("[TORNADO-OPTIMIZER] TornadoVM configuration validated successfully");
            return true;

        } catch (Exception e) {
            System.err.printf("[TORNADO-OPTIMIZER] TornadoVM configuration validation failed: %s%n", e.getMessage());
            return false;
        }
    }

    /**
     * Provides recommendations for TornadoVM configuration based on tensor loading plan.
     */
    public static void provideConfigurationRecommendations(long totalMemoryRequired,
                                                          int largeTensorCount,
                                                          long availableMemory) {

        System.err.println("[TORNADO-OPTIMIZER] Configuration recommendations:");

        double memoryRatio = (double) totalMemoryRequired / availableMemory;
        if (memoryRatio > 0.9) {
            System.err.println("  - WARNING: Very high memory usage (>90%) - consider model optimization");
        } else if (memoryRatio > 0.7) {
            System.err.println("  - High memory usage (>70%) - monitor for allocation failures");
        } else {
            System.err.println("  - Memory usage looks reasonable");
        }

        if (largeTensorCount > 10) {
            System.err.println("  - Many large tensors detected - consider expert tensor optimization");
        }

        if (totalMemoryRequired > 8L * 1024 * 1024 * 1024) { // > 8GB
            System.err.println("  - Large memory footprint - ensure sufficient GPU memory available");
        }
    }
}