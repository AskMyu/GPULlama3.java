package org.beehive.gpullama3.tornadovm;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility to detect OpenCL device memory limits and capabilities.
 *
 * This class automatically detects the maximum allocation size supported by
 * the OpenCL driver, which is crucial for SmartCacheArray batching decisions.
 *
 * Key detection targets:
 * - Max memory allocation (critical for large tensor allocation)
 * - Global memory size (total GPU memory)
 * - Device compute capabilities
 */
public class OpenCLMemoryDetector {
    private static final Logger logger = Logger.getLogger(OpenCLMemoryDetector.class.getName());

    // Cached values (computed once per session)
    private static final AtomicLong maxAllocationSize = new AtomicLong(-1);
    private static final AtomicLong globalMemorySize = new AtomicLong(-1);
    private static volatile boolean detectionAttempted = false;

    // Fallback values for when detection fails
    private static final long FALLBACK_MAX_ALLOCATION = 2L * 1024 * 1024 * 1024; // 2GB safe fallback
    private static final long FALLBACK_GLOBAL_MEMORY = 8L * 1024 * 1024 * 1024;  // 8GB safe fallback

    // Pattern to match memory sizes with units
    private static final Pattern MEMORY_PATTERN = Pattern.compile(
        "\\s*(\\d+)\\s*\\(([0-9.]+)\\s*(GiB|MiB|GB|MB|B)\\).*"
    );

    /**
     * Get the maximum single allocation size supported by OpenCL.
     * This is the critical limit for SmartCacheArray batching decisions.
     *
     * @return Maximum allocation size in bytes
     */
    public static long getMaxAllocationSize() {
        ensureDetectionPerformed();
        long maxAlloc = maxAllocationSize.get();
        return maxAlloc > 0 ? maxAlloc : FALLBACK_MAX_ALLOCATION;
    }

    /**
     * Get the total global memory size of the GPU.
     *
     * @return Total GPU memory in bytes
     */
    public static long getGlobalMemorySize() {
        ensureDetectionPerformed();
        long globalMem = globalMemorySize.get();
        return globalMem > 0 ? globalMem : FALLBACK_GLOBAL_MEMORY;
    }

    /**
     * Get the recommended batch size for SmartCacheArray based on OpenCL limits.
     * Uses 25% of max allocation size as a conservative batch size.
     *
     * @return Recommended batch size in bytes
     */
    public static long getRecommendedBatchSize() {
        long maxAlloc = getMaxAllocationSize();
        // Use 25% of max allocation as batch size for safety margin
        long recommendedSize = maxAlloc / 4;

        // Ensure minimum of 64MB for efficiency
        long minBatchSize = 64L * 1024 * 1024;

        // Ensure maximum of 1GB to avoid large batch overhead
        long maxBatchSize = 1024L * 1024 * 1024;

        return Math.max(minBatchSize, Math.min(recommendedSize, maxBatchSize));
    }

    /**
     * Check if a tensor size requires SmartCacheArray batching.
     *
     * @param tensorSizeBytes Size of tensor in bytes
     * @return true if batching is recommended
     */
    public static boolean requiresBatching(long tensorSizeBytes) {
        long maxAlloc = getMaxAllocationSize();
        // Use 80% of max allocation as threshold for batching decision
        return tensorSizeBytes > (maxAlloc * 0.8);
    }

    /**
     * Get optimal batch count for a given tensor size.
     *
     * @param tensorSizeBytes Size of tensor in bytes
     * @return Recommended number of batches
     */
    public static int getOptimalBatchCount(long tensorSizeBytes) {
        if (!requiresBatching(tensorSizeBytes)) {
            return 1;
        }

        long batchSize = getRecommendedBatchSize();
        return (int) Math.ceil((double) tensorSizeBytes / batchSize);
    }

    /**
     * Perform OpenCL memory detection by parsing clinfo output.
     */
    private static synchronized void ensureDetectionPerformed() {
        if (detectionAttempted) {
            return;
        }

        detectionAttempted = true;

        try {
            logger.info("[OPENCL-DETECT] Starting OpenCL memory capability detection");

            // Execute clinfo command
            Process process = new ProcessBuilder("clinfo").start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String line;
            boolean foundDevice = false;
            long tempMaxAlloc = -1;
            long tempGlobalMem = -1;

            while ((line = reader.readLine()) != null) {
                line = line.trim();

                // Look for memory allocation lines
                if (line.contains("Max memory allocation")) {
                    tempMaxAlloc = parseMemoryValue(line);
                    if (tempMaxAlloc > 0) {
                        logger.info(String.format("[OPENCL-DETECT] Found max allocation: %.2f GB",
                                                tempMaxAlloc / (1024.0 * 1024.0 * 1024.0)));
                    }
                }

                if (line.contains("Global memory size")) {
                    tempGlobalMem = parseMemoryValue(line);
                    if (tempGlobalMem > 0) {
                        logger.info(String.format("[OPENCL-DETECT] Found global memory: %.2f GB",
                                                tempGlobalMem / (1024.0 * 1024.0 * 1024.0)));
                        foundDevice = true;
                    }
                }
            }

            process.waitFor();
            reader.close();

            // Store results if we found valid values
            if (tempMaxAlloc > 0) {
                maxAllocationSize.set(tempMaxAlloc);
                logger.info(String.format("[OPENCL-DETECT] Set max allocation limit: %.2f GB",
                                        tempMaxAlloc / (1024.0 * 1024.0 * 1024.0)));
            }

            if (tempGlobalMem > 0) {
                globalMemorySize.set(tempGlobalMem);
            }

            if (foundDevice) {
                // Calculate and log recommended settings
                long batchSize = getRecommendedBatchSize();
                logger.info(String.format("[OPENCL-DETECT] Recommended SmartCacheArray batch size: %.1f MB",
                                        batchSize / (1024.0 * 1024.0)));
            } else {
                logger.warning("[OPENCL-DETECT] No OpenCL devices found, using fallback values");
            }

        } catch (Exception e) {
            logger.warning("[OPENCL-DETECT] Failed to detect OpenCL capabilities: " + e.getMessage());
            logger.info("[OPENCL-DETECT] Using safe fallback values");
        }
    }

    /**
     * Parse memory value from clinfo output line.
     * Handles formats like "12741951488 (11.87GiB)" or "12741951488 (11.87GB)"
     */
    private static long parseMemoryValue(String line) {
        try {
            Matcher matcher = MEMORY_PATTERN.matcher(line);
            if (matcher.find()) {
                // Try to parse the byte value directly (more accurate)
                String bytesStr = matcher.group(1);
                return Long.parseLong(bytesStr);
            }

            // Fallback: look for standalone numbers
            String[] parts = line.split("\\s+");
            for (String part : parts) {
                if (part.matches("\\d+")) {
                    long value = Long.parseLong(part);
                    // Sanity check: should be a reasonable memory size
                    if (value > 1024 * 1024 && value < Long.MAX_VALUE / 2) {
                        return value;
                    }
                }
            }

        } catch (Exception e) {
            logger.fine("[OPENCL-DETECT] Failed to parse memory value from: " + line);
        }

        return -1;
    }

    /**
     * Get detection status and summary information.
     */
    public static String getDetectionSummary() {
        ensureDetectionPerformed();

        StringBuilder summary = new StringBuilder();
        summary.append("OpenCL Memory Detection Summary:\\n");

        long maxAlloc = maxAllocationSize.get();
        long globalMem = globalMemorySize.get();

        if (maxAlloc > 0) {
            summary.append(String.format("  Max Allocation: %.2f GB\\n", maxAlloc / (1024.0 * 1024.0 * 1024.0)));
        } else {
            summary.append(String.format("  Max Allocation: %.2f GB (fallback)\\n", FALLBACK_MAX_ALLOCATION / (1024.0 * 1024.0 * 1024.0)));
        }

        if (globalMem > 0) {
            summary.append(String.format("  Global Memory: %.2f GB\\n", globalMem / (1024.0 * 1024.0 * 1024.0)));
        } else {
            summary.append(String.format("  Global Memory: %.2f GB (fallback)\\n", FALLBACK_GLOBAL_MEMORY / (1024.0 * 1024.0 * 1024.0)));
        }

        long batchSize = getRecommendedBatchSize();
        summary.append(String.format("  Recommended Batch Size: %.1f MB\\n", batchSize / (1024.0 * 1024.0)));

        return summary.toString();
    }

    /**
     * Force re-detection of OpenCL capabilities (for testing).
     */
    public static synchronized void forceRedetection() {
        detectionAttempted = false;
        maxAllocationSize.set(-1);
        globalMemorySize.set(-1);
        logger.info("[OPENCL-DETECT] Forcing re-detection of OpenCL capabilities");
    }
}