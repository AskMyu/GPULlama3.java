package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.TornadoBackend;
import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.logging.Logger;

/**
 * OpenCL Memory Manager for OLMoE GPU-Only Implementation
 *
 * Detects GPU memory capabilities and selects optimal loading strategy:
 * - Strategy A (Full Loading): Load all 64 experts at startup (24GB+ VRAM)
 * - Strategy B (Dynamic Loading): Load experts on-demand with caching (8-24GB VRAM)
 *
 * Based on OpenCL constraint: CL_DEVICE_MAX_MEM_ALLOC_SIZE typically ~25% of total VRAM
 */
public class OpenCLMemoryManager {
    private static final Logger logger = Logger.getLogger(OpenCLMemoryManager.class.getName());

    // Memory strategy enums
    public enum MemoryStrategy {
        FULL_LOADING,    // Load all experts at startup (optimal)
        DYNAMIC_LOADING, // Load experts on-demand (constrained)
        MINIMAL_LOADING  // Aggressive chunking (minimal memory)
    }

    // Memory detection results
    private long totalGlobalMemory = 0;
    private long maxMemAllocSize = 0;
    private long availableMemory = 0;
    private MemoryStrategy selectedStrategy = MemoryStrategy.DYNAMIC_LOADING;
    private boolean initialized = false;

    // OLMoE model memory requirements
    private static final long EXPERT_WEIGHT_SIZE = 52L * 1024L * 1024L; // ~52MB per expert matrix (worst case)
    private static final long TOTAL_EXPERT_WEIGHTS = 64L * 3L * EXPERT_WEIGHT_SIZE; // 64 experts × 3 matrices
    private static final long BASE_MODEL_MEMORY = 1024L * 1024L * 1024L; // ~1GB base model
    private static final long ACTIVATION_MEMORY = 512L * 1024L * 1024L; // ~512MB activations
    private static final long TOTAL_MODEL_MEMORY = TOTAL_EXPERT_WEIGHTS + BASE_MODEL_MEMORY + ACTIVATION_MEMORY;

    // Cached expert management (for dynamic loading)
    private ExpertCache expertCache = null;
    private int maxCachedExperts = 16; // Default cache size

    /**
     * Initialize memory manager and detect GPU capabilities
     */
    public void initialize() {
        if (initialized) {
            return;
        }

        logger.info("[OLMOE-MEMORY] Initializing OpenCL memory detection...");

        try {
            detectGPUMemoryCapabilities();
            selectOptimalStrategy();
            initializeForStrategy();
            initialized = true;

            logger.info(String.format("[OLMOE-MEMORY] ✅ Initialization complete - Strategy: %s", selectedStrategy));
            logMemoryConfiguration();

        } catch (Exception e) {
            logger.severe("[OLMOE-MEMORY] ❌ Failed to initialize memory manager: " + e.getMessage());
            e.printStackTrace();
            // Fallback to dynamic loading as safe default
            selectedStrategy = MemoryStrategy.DYNAMIC_LOADING;
            initialized = true;
        }
    }

    /**
     * Detect GPU memory capabilities using TornadoVM/OpenCL
     */
    private void detectGPUMemoryCapabilities() {
        try {
            // Get TornadoVM backend and device information
            TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
            TornadoBackend backend = runtime.getBackend(0);
            TornadoDeviceContext deviceContext = backend.getDefaultDevice().getDeviceContext();

            // First, get actual total VRAM from nvidia-smi
            totalGlobalMemory = queryActualVRAM();

            // Then, empirically test OpenCL allocation limits (the real constraint)
            maxMemAllocSize = queryOpenCLMaxAllocation(deviceContext);

            // Available memory (conservative: 85% of total to leave room for OS/driver)
            availableMemory = (long)(totalGlobalMemory * 0.85);

            logger.info(String.format("[OLMOE-MEMORY] GPU Memory Detection:"));
            logger.info(String.format("  Total VRAM: %.2f GB", totalGlobalMemory / (1024.0 * 1024.0 * 1024.0)));
            logger.info(String.format("  OpenCL Max Allocation: %.2f GB", maxMemAllocSize / (1024.0 * 1024.0 * 1024.0)));
            logger.info(String.format("  Available for model: %.2f GB", availableMemory / (1024.0 * 1024.0 * 1024.0)));

        } catch (Exception e) {
            logger.warning("[OLMOE-MEMORY] Could not detect exact GPU memory, using defaults: " + e.getMessage());
            // Conservative defaults for unknown GPUs
            totalGlobalMemory = 8L * 1024L * 1024L * 1024L; // Assume 8GB
            maxMemAllocSize = 2L * 1024L * 1024L * 1024L;   // Assume 2GB max alloc
            availableMemory = 6L * 1024L * 1024L * 1024L;    // Assume 6GB available
        }
    }

    /**
     * Get actual VRAM from nvidia-smi
     */
    private long queryActualVRAM() {
        try {
            ProcessBuilder pb = new ProcessBuilder("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits");
            Process process = pb.start();
            java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()));
            String line = reader.readLine();
            if (line != null && !line.trim().isEmpty()) {
                long vramMB = Long.parseLong(line.trim());
                long vramBytes = vramMB * 1024L * 1024L;
                logger.info(String.format("[OLMOE-MEMORY] nvidia-smi detected: %.2f GB total VRAM",
                                         vramBytes / (1024.0 * 1024.0 * 1024.0)));
                return vramBytes;
            }
        } catch (Exception e) {
            logger.warning("[OLMOE-MEMORY] nvidia-smi failed: " + e.getMessage());
        }
        return 48L * 1024L * 1024L * 1024L; // Default to 48GB
    }

    /**
     * Get actual OpenCL max allocation size from system
     */
    private long queryOpenCLMaxAllocation(TornadoDeviceContext deviceContext) {
        // First try to get the real OpenCL limit using clinfo
        long actualOpenCLLimit = queryOpenCLLimitFromSystem();

        if (actualOpenCLLimit > 0) {
            return actualOpenCLLimit;
        }

        // Fallback: Test FloatArray allocations (limited by Java int constraint)
        long testSize = 1L * 1024L * 1024L * 1024L; // Start with 1GB
        long maxSuccessful = 0;

        logger.info("[OLMOE-MEMORY] Testing TornadoVM FloatArray allocation limits...");

        // Test up to Integer.MAX_VALUE constraint (FloatArray limitation ~8GB)
        for (int i = 0; i < 10; i++) {
            try {
                int elements = (int)(testSize / 4);
                if (elements <= 0 || elements > Integer.MAX_VALUE - 1000) {
                    logger.info(String.format("[OLMOE-MEMORY] Hit FloatArray integer limit at %.2f GB",
                                             testSize / (1024.0 * 1024.0 * 1024.0)));
                    break;
                }

                FloatArray testArray = new FloatArray(elements);
                testArray.set(0, 1.0f); // Touch memory to ensure allocation
                testArray.set(elements - 1, 1.0f); // Touch end
                maxSuccessful = testSize;

                testSize += 1L * 1024L * 1024L * 1024L; // Add 1GB
            } catch (OutOfMemoryError | Exception e) {
                logger.info(String.format("[OLMOE-MEMORY] ❌ FloatArray failed at %.2f GB: %s",
                                         testSize / (1024.0 * 1024.0 * 1024.0), e.getClass().getSimpleName()));
                break;
            }
        }

        if (maxSuccessful == 0) {
            logger.warning("[OLMOE-MEMORY] No successful allocations, using 2GB default");
            return 2L * 1024L * 1024L * 1024L;
        }

        logger.warning(String.format("[OLMOE-MEMORY] Using FloatArray limit: %.2f GB (not true OpenCL limit)",
                                     maxSuccessful / (1024.0 * 1024.0 * 1024.0)));
        return maxSuccessful;
    }

    /**
     * Query actual OpenCL max allocation size from clinfo
     */
    private long queryOpenCLLimitFromSystem() {
        try {
            ProcessBuilder pb = new ProcessBuilder("clinfo");
            Process process = pb.start();
            java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()));
            String line;

            while ((line = reader.readLine()) != null) {
                if (line.toLowerCase().contains("max memory allocation") && line.contains("(")) {
                    // Parse line like: "  Max memory allocation                           12741951488 (11.87GiB)"
                    String[] parts = line.trim().split("\\s+");
                    for (String part : parts) {
                        if (part.matches("\\d+") && part.length() > 8) { // Large number in bytes
                            long bytes = Long.parseLong(part);
                            if (bytes > 1024L * 1024L * 1024L) { // > 1GB, reasonable
                                logger.info(String.format("[OLMOE-MEMORY] clinfo detected OpenCL max allocation: %.2f GB",
                                                         bytes / (1024.0 * 1024.0 * 1024.0)));
                                return bytes;
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            logger.fine("[OLMOE-MEMORY] clinfo not available: " + e.getMessage());
        }

        // Fallback: Use conservative estimate (25% of total VRAM)
        long estimate = totalGlobalMemory / 4;
        logger.info(String.format("[OLMOE-MEMORY] Using estimated OpenCL limit (25%% of VRAM): %.2f GB",
                                 estimate / (1024.0 * 1024.0 * 1024.0)));
        return estimate;
    }


    /**
     * Select optimal memory strategy based on detected capabilities
     */
    private void selectOptimalStrategy() {
        logger.info(String.format("[OLMOE-MEMORY] Selecting strategy for model size: %.2f GB",
                                 TOTAL_MODEL_MEMORY / (1024.0 * 1024.0 * 1024.0)));

        // Check if we can load all experts in a single allocation
        boolean canLoadAllExperts = (TOTAL_EXPERT_WEIGHTS < maxMemAllocSize);

        // Check if total model fits in available memory
        boolean modelFitsInMemory = (TOTAL_MODEL_MEMORY < availableMemory);

        if (canLoadAllExperts && modelFitsInMemory && totalGlobalMemory >= 24L * 1024L * 1024L * 1024L) {
            // Tier 1: Full loading (24GB+ VRAM)
            selectedStrategy = MemoryStrategy.FULL_LOADING;
            logger.info("[OLMOE-MEMORY] ✅ Selected FULL_LOADING strategy (all experts in memory)");

        } else if (totalGlobalMemory >= 8L * 1024L * 1024L * 1024L) {
            // Tier 2: Dynamic loading (8-24GB VRAM)
            selectedStrategy = MemoryStrategy.DYNAMIC_LOADING;

            // Calculate optimal cache size
            long cacheMemory = availableMemory - BASE_MODEL_MEMORY - ACTIVATION_MEMORY;
            maxCachedExperts = (int) Math.min(32, cacheMemory / (3L * EXPERT_WEIGHT_SIZE));
            maxCachedExperts = Math.max(8, maxCachedExperts); // At least 8 experts cached

            logger.info(String.format("[OLMOE-MEMORY] ✅ Selected DYNAMIC_LOADING strategy (cache %d experts)",
                                     maxCachedExperts));

        } else {
            // Tier 3: Minimal loading (<8GB VRAM)
            selectedStrategy = MemoryStrategy.MINIMAL_LOADING;
            maxCachedExperts = 4; // Very limited caching
            logger.info("[OLMOE-MEMORY] ⚠️ Selected MINIMAL_LOADING strategy (limited memory)");
        }
    }

    /**
     * Initialize resources for selected strategy
     */
    private void initializeForStrategy() {
        switch (selectedStrategy) {
            case FULL_LOADING:
                // No special initialization needed - weights loaded at model load time
                break;

            case DYNAMIC_LOADING:
            case MINIMAL_LOADING:
                // Initialize expert cache for dynamic loading
                expertCache = new ExpertCache(maxCachedExperts);
                logger.info(String.format("[OLMOE-MEMORY] Initialized expert cache with capacity: %d",
                                         maxCachedExperts));
                break;
        }
    }

    /**
     * Log memory configuration for debugging
     */
    public void logMemoryConfiguration() {
        logger.info("╔══════════════════════════════════════════════════════════╗");
        logger.info("║           OLMoE Memory Configuration Summary              ║");
        logger.info("╠══════════════════════════════════════════════════════════╣");
        logger.info(String.format("║ Total VRAM:           %8.2f GB                        ║",
                                 totalGlobalMemory / (1024.0 * 1024.0 * 1024.0)));
        logger.info(String.format("║ Max Allocation:       %8.2f GB                        ║",
                                 maxMemAllocSize / (1024.0 * 1024.0 * 1024.0)));
        logger.info(String.format("║ Available Memory:     %8.2f GB                        ║",
                                 availableMemory / (1024.0 * 1024.0 * 1024.0)));
        logger.info(String.format("║ Model Requirements:   %8.2f GB                        ║",
                                 TOTAL_MODEL_MEMORY / (1024.0 * 1024.0 * 1024.0)));
        logger.info(String.format("║ Strategy:             %-20s               ║", selectedStrategy));
        if (selectedStrategy != MemoryStrategy.FULL_LOADING) {
            logger.info(String.format("║ Expert Cache Size:    %3d experts                         ║",
                                     maxCachedExperts));
        }
        logger.info("╚══════════════════════════════════════════════════════════╝");
    }

    // Getter methods
    public boolean isInitialized() { return initialized; }
    public MemoryStrategy getStrategy() { return selectedStrategy; }
    public ExpertCache getExpertCache() { return expertCache; }
    public boolean canLoadAllExperts() { return selectedStrategy == MemoryStrategy.FULL_LOADING; }
    public long getTotalMemory() { return totalGlobalMemory; }
    public long getMaxAllocSize() { return maxMemAllocSize; }
    public long getAvailableMemory() { return availableMemory; }

    /**
     * Log current memory usage statistics
     */
    public void logMemoryUsage(long usedMemory) {
        double usedGB = usedMemory / (1024.0 * 1024.0 * 1024.0);
        double totalGB = totalGlobalMemory / (1024.0 * 1024.0 * 1024.0);
        double percentage = (usedMemory * 100.0) / totalGlobalMemory;

        logger.info(String.format("[OLMOE-MEMORY] Memory usage: %.2f GB / %.2f GB (%.1f%%)",
                                 usedGB, totalGB, percentage));

        if (expertCache != null) {
            expertCache.logCacheStatistics();
        }
    }
}