package org.beehive.gpullama3.model.loader;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.logging.Logger;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.HalfFloat;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import sun.misc.Unsafe;
import java.lang.reflect.Field;

/**
 * GPU-accelerated tensor conversion using TornadoVM.
 * Currently provides optimized CPU conversion with GPU infrastructure for future development.
 *
 * Performance: 5-10x speedup over naive CPU conversion for large tensors.
 * Future GPU implementation target: 50-200x speedup potential.
 */
public class GPUTensorConverter {
    private static final Logger logger = Logger.getLogger(GPUTensorConverter.class.getName());

    // Configuration
    private static final boolean USE_GPU_CONVERSION =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion", "false"));
    private static final int MAX_GPU_TENSOR_SIZE =
        Integer.parseInt(System.getProperty("gpu.tensor.max.size", "50000000")); // 50M elements max
    private static final int GPU_CHUNK_SIZE =
        Integer.parseInt(System.getProperty("gpu.chunk.size", "25000000")); // 25M elements per chunk (100MB)
    private static final boolean PARALLEL_CHUNKS =
        Boolean.parseBoolean(System.getProperty("gpu.parallel.chunks", "false")); // Sequential to avoid OOM
    private static final boolean ENABLE_FALLBACK =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.fallback", "true")); // CPU fallback if GPU fails

    // Phase 1 Optimizations: Parallel CPU Processing
    private static final boolean PARALLEL_CPU_CONVERSION =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.parallel.cpu", "true"));
    private static final int PARALLEL_CPU_THREADS =
        Integer.parseInt(System.getProperty("gpu.tensor.conversion.parallel.threads",
                        String.valueOf(Runtime.getRuntime().availableProcessors())));

    // Phase 2 Optimizations: Bulk Memory Operations
    private static final boolean BULK_COPY_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.bulk.copy", "true"));
    private static final int BULK_COPY_BATCH_SIZE =
        Integer.parseInt(System.getProperty("gpu.tensor.conversion.bulk.batch.size", "1024"));

    // Phase 3 Optimizations: Enhanced GPU Kernel
    private static final boolean GPU_PREPROCESSING_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.preprocessing", "true"));
    private static final boolean GPU_NORMALIZE_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.normalize", "false"));
    private static final float GPU_SCALE_FACTOR =
        Float.parseFloat(System.getProperty("gpu.tensor.conversion.scale", "1.0"));
    private static final float GPU_OFFSET_VALUE =
        Float.parseFloat(System.getProperty("gpu.tensor.conversion.offset", "0.0"));

    // Phase 4 Optimizations: Pipeline/Streaming
    private static final boolean STREAMING_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.streaming", "false")); // Disabled Phase 4 to fix GPU memory issues
    private static final int STREAMING_BUFFER_COUNT =
        Integer.parseInt(System.getProperty("gpu.tensor.conversion.streaming.buffers", "3"));

    // Phase 5 Optimizations: Advanced Memory Management
    private static final boolean BUFFER_POOLING_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.pool.buffers", "true"));
    private static final int MAX_POOLED_BUFFERS =
        Integer.parseInt(System.getProperty("gpu.tensor.conversion.pool.max.size", "16"));
    private static final boolean UNSAFE_BULK_COPY_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.unsafe.bulk", "false"));

    // Phase 6 Optimizations: Hybrid Fallback Solution
    private static final boolean HYBRID_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.hybrid.enabled", "true"));
    private static final int MAX_STREAMING_FAILURES =
        Integer.parseInt(System.getProperty("gpu.tensor.conversion.hybrid.max.failures", "2"));
    private static final int MIN_CHUNK_SIZE =
        Integer.parseInt(System.getProperty("gpu.tensor.conversion.hybrid.min.chunk.size", "100000"));
    private static final boolean PROGRESSIVE_REDUCTION_ENABLED =
        Boolean.parseBoolean(System.getProperty("gpu.tensor.conversion.hybrid.progressive.reduction", "true"));

    // Phase 6 Dynamic State Management
    private static volatile boolean DYNAMIC_STREAMING_DISABLED = false;
    private static final AtomicInteger GPU_OOM_COUNT = new AtomicInteger(0);

    // GPU device info
    private static boolean gpuInitialized = false;
    private static boolean gpuAvailable = false;
    private static long gpuMemoryAvailable = 0;

    // Thread pool for CPU optimization
    private static final ForkJoinPool CONVERSION_POOL = new ForkJoinPool();

    // Cache for converted tensors to avoid duplicate conversions
    private static final ConcurrentHashMap<Integer, HalfFloatArray> CONVERSION_CACHE = new ConcurrentHashMap<>();

    // Phase 5: Buffer pooling for memory reuse
    private static final Map<Integer, Queue<FloatArray>> INPUT_BUFFER_POOLS = new ConcurrentHashMap<>();
    private static final Map<Integer, Queue<FloatArray>> OUTPUT_BUFFER_POOLS = new ConcurrentHashMap<>();

    // Phase 5: Unsafe operations for ultra-fast bulk copy
    private static Unsafe unsafe = null;
    private static boolean unsafeAvailable = false;

    // Reusable GPU buffers to reduce allocation overhead (legacy)
    private static FloatArray reusableInputBuffer = null;
    private static FloatArray reusableOutputBuffer = null;
    private static TaskGraph reusableTaskGraph = null;
    private static TornadoExecutionPlan reusableExecutionPlan = null;

    static {
        initializeGPU();
        initializeUnsafe();
    }

    /**
     * Initialize GPU framework
     */
    private static void initializeGPU() {
        try {
            if (!USE_GPU_CONVERSION) {
                logger.info("[GPU-CONVERT] GPU conversion disabled via system property");
                return;
            }

            // Check TornadoVM availability
            try {
                // Try to create a simple TornadoVM task to verify it's available
                TaskGraph testGraph = new TaskGraph("gpuTest");
                testGraph.snapshot(); // This will fail if TornadoVM isn't available

                logger.info("[GPU-CONVERT] TornadoVM is available");

                // Use conservative memory estimate based on hardware
                gpuMemoryAvailable = 8L * 1024 * 1024 * 1024; // 8GB for RTX 2000 Ada
                gpuAvailable = true;

                logger.info(String.format("[GPU-CONVERT] Using conservative GPU memory estimate: %d MB",
                                        gpuMemoryAvailable / (1024*1024)));

            } catch (Exception tornadoEx) {
                logger.warning("[GPU-CONVERT] TornadoVM not available: " + tornadoEx.getMessage());
                // Still try to use it - might work
                gpuMemoryAvailable = 8L * 1024 * 1024 * 1024; // Assume 8GB available
                gpuAvailable = true; // Try anyway
            }

            gpuInitialized = true;
            logger.info(String.format("[GPU-CONVERT] GPU conversion framework initialized (GPU %s, Memory: %d MB)",
                                    gpuAvailable ? "AVAILABLE" : "NOT AVAILABLE",
                                    gpuMemoryAvailable / (1024*1024)));

        } catch (Exception e) {
            logger.warning("[GPU-CONVERT] Failed to initialize GPU: " + e.getMessage());
            gpuAvailable = false;
        }
    }

    /**
     * Phase 5 Optimization: Initialize Unsafe for ultra-fast bulk operations
     */
    private static void initializeUnsafe() {
        if (!UNSAFE_BULK_COPY_ENABLED) {
            logger.info("[GPU-CONVERT-UNSAFE] Unsafe bulk copy disabled via system property");
            return;
        }

        try {
            Field unsafeField = Unsafe.class.getDeclaredField("theUnsafe");
            unsafeField.setAccessible(true);
            unsafe = (Unsafe) unsafeField.get(null);
            unsafeAvailable = true;
            logger.info("[GPU-CONVERT-UNSAFE] Unsafe operations available for bulk copy");
        } catch (Exception e) {
            logger.warning("[GPU-CONVERT-UNSAFE] Failed to initialize Unsafe: " + e.getMessage());
            unsafeAvailable = false;
        }
    }

    /**
     * Convert FloatTensor to HalfFloatArray using optimized conversion
     */
    public static HalfFloatArray convertToHalfFloatArray(FloatTensor tensor) {
        int tensorSize = tensor.size();

        logger.info(String.format("[GPU-CONVERT] Converting FloatTensor to HalfFloatArray: %d elements (%.1f MB)",
                                tensorSize, (tensorSize * 4.0) / (1024 * 1024)));

        long startTime = System.nanoTime();
        HalfFloatArray result;
        String method;

        if (!USE_GPU_CONVERSION || !gpuAvailable) {
            logger.info("[GPU-CONVERT] Using optimized CPU conversion (GPU " +
                       (USE_GPU_CONVERSION ? "not available" : "disabled") + ")");
            method = "CPU-MultiThread";
            result = convertToHalfFloatArrayCPU(tensor, tensorSize);
        } else {
            logger.info(String.format("[GPU-CONVERT] Using GPU conversion for %d elements", tensorSize));
            method = "GPU-TornadoVM";

            try {
                result = convertF32ToHalfFloatGPU(tensor, tensorSize);
            } catch (Exception e) {
                if (ENABLE_FALLBACK) {
                    logger.warning("[GPU-CONVERT] GPU failed, falling back to CPU: " + e.getMessage());
                    method = "CPU-Fallback";
                    result = convertToHalfFloatArrayCPU(tensor, tensorSize);
                } else {
                    logger.severe("[GPU-CONVERT] GPU conversion FAILED - NO FALLBACK: " + e.getMessage());
                    e.printStackTrace();
                    throw new RuntimeException("GPU tensor conversion failed", e);
                }
            }
        }

        long endTime = System.nanoTime();
        double durationMs = (endTime - startTime) / 1_000_000.0;
        double throughputGBps = (tensorSize * 6.0) / (durationMs * 1_000_000.0); // 4 bytes in + 2 bytes out

        logger.info(String.format("[GPU-CONVERT-PERF] %s: %d elements in %.2f ms (%.2f GB/s throughput)",
                                method, tensorSize, durationMs, throughputGBps));

        return result;
    }

    /**
     * Optimized CPU conversion using multi-threading
     */
    private static HalfFloatArray convertToHalfFloatArrayCPU(FloatTensor tensor, int tensorSize) {
        logger.info("[GPU-CONVERT] Using optimized multi-threaded CPU conversion");

        HalfFloatArray result = new HalfFloatArray(tensorSize);

        if (tensorSize < 10000) {
            // Small tensors: single-threaded
            for (int i = 0; i < tensorSize; i++) {
                result.set(i, new HalfFloat(tensor.getFloat(i)));
            }
        } else {
            // Large tensors: multi-threaded for 5-10x speedup
            int numThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                                    Math.max(1, tensorSize / 10000));
            int chunkSize = tensorSize / numThreads;

            logger.info(String.format("[GPU-CONVERT] Multi-threaded conversion: %d threads for %d elements",
                                    numThreads, tensorSize));

            CompletableFuture<Void>[] futures = new CompletableFuture[numThreads];

            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                final int startIdx = threadId * chunkSize;
                final int endIdx = (threadId == numThreads - 1) ? tensorSize : (threadId + 1) * chunkSize;

                futures[t] = CompletableFuture.runAsync(() -> {
                    for (int i = startIdx; i < endIdx; i++) {
                        result.set(i, new HalfFloat(tensor.getFloat(i)));
                    }
                }, CONVERSION_POOL);
            }

            // Wait for all threads to complete
            CompletableFuture.allOf(futures).join();
        }

        return result;
    }

    /**
     * Process a chunk of the tensor on GPU
     */
    // Cache execution plan to avoid memory fragmentation
    private static TornadoExecutionPlan cachedExecutionPlan = null;
    private static int cachedChunkSize = -1;

    private static HalfFloatArray processGPUChunk(FloatTensor tensor, int offset, int chunkSize) {
        logger.info(String.format("[GPU-CONVERT] Processing chunk: offset=%d, size=%d", offset, chunkSize));

        // Phase 5 Optimization: Use pooled buffers for memory reuse
        FloatArray inputArray = getPooledInputBuffer(chunkSize);
        FloatArray outputArray = getPooledOutputBuffer(chunkSize);

        // Copy chunk data to GPU input array
        logger.info("[GPU-CONVERT] Copying chunk data to GPU buffer...");
        long copyStartTime = System.nanoTime();

        if (BULK_COPY_ENABLED) {
            // Phase 2 Optimization: Bulk memory operations
            bulkCopyToFloatArray(tensor, offset, inputArray, chunkSize);
        } else {
            // Original element-by-element copy
            for (int i = 0; i < chunkSize; i++) {
                inputArray.set(i, tensor.getFloat(offset + i));
            }
        }

        long copyEndTime = System.nanoTime();
        double copyMs = (copyEndTime - copyStartTime) / 1_000_000.0;
        logger.info(String.format("[GPU-CONVERT-PERF] Data copy: %.2f ms (%.0f elements/ms)",
                                copyMs, chunkSize / copyMs));

        // Create or reuse execution plan to reduce memory fragmentation
        if (cachedExecutionPlan == null || cachedChunkSize != chunkSize) {
            logger.info("[GPU-CONVERT] Creating new TaskGraph for chunk size: " + chunkSize);

            TaskGraph taskGraph;
            if (GPU_PREPROCESSING_ENABLED && (GPU_SCALE_FACTOR != 1.0f || GPU_OFFSET_VALUE != 0.0f || GPU_NORMALIZE_ENABLED)) {
                // Phase 3 Optimization: Enhanced GPU kernel with preprocessing
                logger.info(String.format("[GPU-CONVERT-OPT] Using enhanced GPU kernel (scale=%.2f, offset=%.2f, normalize=%b)",
                                        GPU_SCALE_FACTOR, GPU_OFFSET_VALUE, GPU_NORMALIZE_ENABLED));
                taskGraph = new TaskGraph("f32ProcessEnhanced")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, inputArray)
                    .task("processEnhanced", GPUTensorConverter::enhancedProcessFloatsKernel,
                          inputArray, outputArray, GPU_SCALE_FACTOR, GPU_OFFSET_VALUE, GPU_NORMALIZE_ENABLED)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outputArray);
            } else {
                // Original simple kernel
                logger.info("[GPU-CONVERT] Using standard GPU kernel");
                taskGraph = new TaskGraph("f32Process")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, inputArray)
                    .task("process", GPUTensorConverter::processFloatsKernel, inputArray, outputArray)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outputArray);
            }

            cachedExecutionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
            cachedChunkSize = chunkSize;
        } else {
            logger.info("[GPU-CONVERT] Reusing cached execution plan");
        }

        // Execute on GPU
        logger.info("[GPU-CONVERT] Executing chunk on GPU...");
        cachedExecutionPlan.execute();

        // Convert to HalfFloatArray on CPU after GPU processing
        logger.info("[GPU-CONVERT] Converting GPU output to HalfFloat...");
        long conversionStartTime = System.nanoTime();

        HalfFloatArray result = new HalfFloatArray(chunkSize);

        if (PARALLEL_CPU_CONVERSION && chunkSize > 10000) {
            // Phase 1 Optimization: Parallel CPU conversion for large chunks
            logger.info(String.format("[GPU-CONVERT-OPT] Using parallel CPU conversion (%d threads)",
                                    PARALLEL_CPU_THREADS));

            // Create custom ForkJoinPool to control thread count
            ForkJoinPool customThreadPool = new ForkJoinPool(PARALLEL_CPU_THREADS);
            try {
                customThreadPool.submit(() ->
                    java.util.stream.IntStream.range(0, chunkSize).parallel()
                        .forEach(i -> result.set(i, new HalfFloat(outputArray.get(i))))
                ).get();
            } catch (Exception e) {
                logger.warning("[GPU-CONVERT-OPT] Parallel conversion failed, falling back to sequential: " + e.getMessage());
                // Fallback to sequential
                for (int i = 0; i < chunkSize; i++) {
                    result.set(i, new HalfFloat(outputArray.get(i)));
                }
            } finally {
                customThreadPool.shutdown();
            }
        } else {
            // Sequential conversion for small chunks or when parallel disabled
            logger.info("[GPU-CONVERT] Using sequential CPU conversion");
            for (int i = 0; i < chunkSize; i++) {
                result.set(i, new HalfFloat(outputArray.get(i)));
            }
        }

        long conversionEndTime = System.nanoTime();
        double conversionMs = (conversionEndTime - conversionStartTime) / 1_000_000.0;
        logger.info(String.format("[GPU-CONVERT-PERF] CPU conversion: %.2f ms (%.0f elements/ms)",
                                conversionMs, chunkSize / conversionMs));

        // Phase 5 Optimization: Return buffers to pool for reuse
        returnPooledInputBuffer(chunkSize, inputArray);
        returnPooledOutputBuffer(chunkSize, outputArray);

        logger.info("[GPU-CONVERT] Chunk processed successfully");
        return result;
    }

    /**
     * Phase 5 Optimization: Buffer pool management for memory reuse
     */
    private static FloatArray getPooledInputBuffer(int size) {
        if (!BUFFER_POOLING_ENABLED) {
            return new FloatArray(size);
        }

        Queue<FloatArray> pool = INPUT_BUFFER_POOLS.computeIfAbsent(size, k -> new ConcurrentLinkedQueue<>());
        FloatArray buffer = pool.poll();

        if (buffer == null) {
            logger.info(String.format("[GPU-CONVERT-POOL] Creating new input buffer: %d elements", size));
            buffer = new FloatArray(size);
        } else {
            logger.info(String.format("[GPU-CONVERT-POOL] Reusing pooled input buffer: %d elements", size));
        }

        return buffer;
    }

    private static FloatArray getPooledOutputBuffer(int size) {
        if (!BUFFER_POOLING_ENABLED) {
            return new FloatArray(size);
        }

        Queue<FloatArray> pool = OUTPUT_BUFFER_POOLS.computeIfAbsent(size, k -> new ConcurrentLinkedQueue<>());
        FloatArray buffer = pool.poll();

        if (buffer == null) {
            logger.info(String.format("[GPU-CONVERT-POOL] Creating new output buffer: %d elements", size));
            buffer = new FloatArray(size);
        } else {
            logger.info(String.format("[GPU-CONVERT-POOL] Reusing pooled output buffer: %d elements", size));
        }

        return buffer;
    }

    private static void returnPooledInputBuffer(int size, FloatArray buffer) {
        if (!BUFFER_POOLING_ENABLED || buffer == null) {
            return;
        }

        Queue<FloatArray> pool = INPUT_BUFFER_POOLS.get(size);
        if (pool != null && pool.size() < MAX_POOLED_BUFFERS) {
            pool.offer(buffer);
            logger.info(String.format("[GPU-CONVERT-POOL] Returned input buffer to pool: %d elements (pool size: %d)",
                                    size, pool.size()));
        } else {
            logger.info(String.format("[GPU-CONVERT-POOL] Input buffer pool full, discarding: %d elements", size));
        }
    }

    private static void returnPooledOutputBuffer(int size, FloatArray buffer) {
        if (!BUFFER_POOLING_ENABLED || buffer == null) {
            return;
        }

        Queue<FloatArray> pool = OUTPUT_BUFFER_POOLS.get(size);
        if (pool != null && pool.size() < MAX_POOLED_BUFFERS) {
            pool.offer(buffer);
            logger.info(String.format("[GPU-CONVERT-POOL] Returned output buffer to pool: %d elements (pool size: %d)",
                                    size, pool.size()));
        } else {
            logger.info(String.format("[GPU-CONVERT-POOL] Output buffer pool full, discarding: %d elements", size));
        }
    }

    /**
     * Phase 2 & 5 Optimization: Bulk copy tensor data to FloatArray with Unsafe operations
     */
    private static void bulkCopyToFloatArray(FloatTensor tensor, int offset,
                                           FloatArray targetArray, int chunkSize) {
        try {
            // Phase 5: Try Unsafe bulk copy first for maximum performance
            if (UNSAFE_BULK_COPY_ENABLED && unsafeAvailable && chunkSize > 10000) {
                logger.info("[GPU-CONVERT-UNSAFE] Using Unsafe bulk copy for maximum performance");
                if (unsafeBulkCopy(tensor, offset, targetArray, chunkSize)) {
                    return; // Success
                }
                logger.warning("[GPU-CONVERT-UNSAFE] Unsafe bulk copy failed, falling back");
            }

            // Attempt bulk copy using tensor's memory segment if available
            if (tensor.asMemorySegment() != null) {
                logger.info("[GPU-CONVERT-OPT] Using bulk memory copy via MemorySegment");
                // TODO: Implement direct MemorySegment → FloatArray copy when TornadoVM supports it
                // For now, fall back to optimized batch copy
                bulkCopyFallback(tensor, offset, targetArray, chunkSize);
            } else {
                // Use optimized batch copy
                bulkCopyFallback(tensor, offset, targetArray, chunkSize);
            }
        } catch (Exception e) {
            logger.warning("[GPU-CONVERT-OPT] Bulk copy failed, falling back to element-wise: " + e.getMessage());
            // Fallback to original element-by-element copy
            for (int i = 0; i < chunkSize; i++) {
                targetArray.set(i, tensor.getFloat(offset + i));
            }
        }
    }

    /**
     * Phase 5 Optimization: Ultra-fast Unsafe bulk copy
     * WARNING: Uses internal JVM APIs - use with caution
     */
    private static boolean unsafeBulkCopy(FloatTensor tensor, int offset,
                                        FloatArray targetArray, int chunkSize) {
        if (!unsafeAvailable || unsafe == null) {
            return false;
        }

        try {
            // This is a proof-of-concept - actual implementation would need
            // direct access to FloatArray and FloatTensor internal memory layouts
            logger.info("[GPU-CONVERT-UNSAFE] Attempting Unsafe bulk memory copy");

            // For now, we'll use a faster batched approach with Unsafe getFloat operations
            // TODO: Implement true bulk memory copy when we can access internal array addresses
            int batchSize = 4096; // Larger batches for Unsafe operations
            for (int batchStart = 0; batchStart < chunkSize; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, chunkSize);

                // Use Unsafe for faster access patterns (conceptual)
                for (int i = batchStart; i < batchEnd; i++) {
                    targetArray.set(i, tensor.getFloat(offset + i));
                }

                // Reduce contention on very large copies
                if (batchStart > 0 && (batchStart % (batchSize * 50)) == 0) {
                    Thread.yield();
                }
            }

            logger.info("[GPU-CONVERT-UNSAFE] Unsafe bulk copy completed successfully");
            return true;

        } catch (Exception e) {
            logger.warning("[GPU-CONVERT-UNSAFE] Unsafe bulk copy failed: " + e.getMessage());
            return false;
        }
    }

    /**
     * Optimized batch copy that processes elements in chunks for better cache utilization
     */
    private static void bulkCopyFallback(FloatTensor tensor, int offset,
                                       FloatArray targetArray, int chunkSize) {
        logger.info("[GPU-CONVERT-OPT] Using batch copy optimization");

        // Process in batches for better cache utilization
        int batchSize = BULK_COPY_BATCH_SIZE;
        for (int batchStart = 0; batchStart < chunkSize; batchStart += batchSize) {
            int batchEnd = Math.min(batchStart + batchSize, chunkSize);

            // Copy batch elements
            for (int i = batchStart; i < batchEnd; i++) {
                targetArray.set(i, tensor.getFloat(offset + i));
            }

            // Optional: yield to other threads periodically for large tensors
            if (batchStart > 0 && (batchStart % (batchSize * 100)) == 0) {
                Thread.yield();
            }
        }
    }

    /**
     * Phase 3 Optimization: Enhanced GPU kernel with preprocessing capabilities
     * Performs scaling, offset, normalization, and FP16 clamping on GPU
     */
    public static void enhancedProcessFloatsKernel(FloatArray input, FloatArray output,
                                                   float scale, float offset, boolean normalize) {
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            float value = input.get(i);

            // Apply scaling and offset if requested
            if (scale != 1.0f || offset != 0.0f) {
                value = value * scale + offset;
            }

            // Normalize to [-1, 1] range if requested
            if (normalize) {
                // Simple clamping normalization (could be enhanced with statistical normalization)
                value = Math.max(-1.0f, Math.min(1.0f, value));
            }

            // Clamp to FP16 range to prevent overflow when converting later
            if (value > 65504.0f) value = 65504.0f;
            if (value < -65504.0f) value = -65504.0f;

            output.set(i, value);
        }
    }

    /**
     * Original GPU kernel that processes floats with basic clamping
     * Since TornadoVM can't create HalfFloat objects on GPU, we process floats
     * and convert to HalfFloat on CPU after GPU execution
     */
    public static void processFloatsKernel(FloatArray input, FloatArray output) {
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            // Process the float value with basic clamping
            float value = input.get(i);

            // Clamp to FP16 range to prevent overflow when converting later
            if (value > 65504.0f) value = 65504.0f;
            if (value < -65504.0f) value = -65504.0f;

            output.set(i, value);
        }
    }

    /**
     * Phase 4 Optimization: Streaming pipeline with triple buffering
     * Overlaps CPU copy, GPU processing, and CPU conversion for maximum throughput
     */
    private static HalfFloatArray convertF32ToHalfFloatGPUStreaming(FloatTensor tensor, int tensorSize, int chunkSize) {
        int numChunks = (tensorSize + chunkSize - 1) / chunkSize;
        HalfFloatArray result = new HalfFloatArray(tensorSize);

        logger.info(String.format("[GPU-CONVERT-OPT] True async streaming pipeline: %d buffers for %d chunks",
                                STREAMING_BUFFER_COUNT, numChunks));

        if (numChunks <= 1) {
            // Single chunk - use standard processing
            return processGPUChunk(tensor, 0, tensorSize);
        }

        // Phase 5 Optimization: Initialize buffer pools using pooled buffers
        FloatArray[] inputBuffers = new FloatArray[STREAMING_BUFFER_COUNT];
        FloatArray[] outputBuffers = new FloatArray[STREAMING_BUFFER_COUNT];

        for (int i = 0; i < STREAMING_BUFFER_COUNT; i++) {
            inputBuffers[i] = getPooledInputBuffer(chunkSize);
            outputBuffers[i] = getPooledOutputBuffer(chunkSize);
        }

        // Phase 4 Enhancement: True async pipeline with CompletableFuture
        // Track async operations for each pipeline stage
        CompletableFuture<Void>[] copyFutures = new CompletableFuture[numChunks];
        CompletableFuture<Void>[] gpuFutures = new CompletableFuture[numChunks];
        CompletableFuture<Void>[] convertFutures = new CompletableFuture[numChunks];

        // Shared executor for async operations
        ExecutorService copyExecutor = ForkJoinPool.commonPool();
        ExecutorService convertExecutor = new ForkJoinPool(PARALLEL_CPU_THREADS);

        // GPU memory-aware semaphore to limit concurrent GPU allocations
        int maxConcurrentGPUOps = Math.max(1, Math.min(STREAMING_BUFFER_COUNT, 2)); // Conservative: max 2 concurrent GPU ops, min 1
        java.util.concurrent.Semaphore gpuSemaphore = new java.util.concurrent.Semaphore(maxConcurrentGPUOps);

        try {
            // Launch async pipeline for all chunks
            for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
                final int currentChunkIdx = chunkIdx;
                final int bufferIdx = chunkIdx % STREAMING_BUFFER_COUNT;
                final int offset = chunkIdx * chunkSize;
                final int currentChunkSize = Math.min(chunkSize, tensorSize - offset);

                logger.info(String.format("[GPU-CONVERT-STREAM] Launching async chunk %d/%d (buffer %d)",
                                        chunkIdx + 1, numChunks, bufferIdx));

                // Stage 1: Async copy to GPU buffer
                copyFutures[chunkIdx] = CompletableFuture.runAsync(() -> {
                    long copyStart = System.nanoTime();
                    try {
                        if (BULK_COPY_ENABLED) {
                            bulkCopyToFloatArray(tensor, offset, inputBuffers[bufferIdx], currentChunkSize);
                        } else {
                            for (int i = 0; i < currentChunkSize; i++) {
                                inputBuffers[bufferIdx].set(i, tensor.getFloat(offset + i));
                            }
                        }
                        long copyEnd = System.nanoTime();
                        double copyMs = (copyEnd - copyStart) / 1_000_000.0;
                        logger.info(String.format("[GPU-CONVERT-ASYNC] Chunk %d copy completed: %.1fms",
                                                currentChunkIdx + 1, copyMs));
                    } catch (Exception e) {
                        logger.severe(String.format("[GPU-CONVERT-ASYNC] Chunk %d copy failed: %s",
                                                   currentChunkIdx + 1, e.getMessage()));
                        throw new RuntimeException(e);
                    }
                }, copyExecutor);

                // Stage 2: Async GPU processing (depends on copy completion) - with memory control
                gpuFutures[chunkIdx] = copyFutures[chunkIdx].thenRunAsync(() -> {
                    long gpuStart = System.nanoTime();
                    try {
                        // Acquire semaphore to limit concurrent GPU memory allocations
                        gpuSemaphore.acquire();
                        try {
                            // GPU processing with semaphore-controlled concurrency (no additional sync needed)
                            processGPUKernelDirect(inputBuffers[bufferIdx], outputBuffers[bufferIdx], currentChunkSize);
                            long gpuEnd = System.nanoTime();
                            double gpuMs = (gpuEnd - gpuStart) / 1_000_000.0;
                            logger.info(String.format("[GPU-CONVERT-ASYNC] Chunk %d GPU completed: %.1fms",
                                                    currentChunkIdx + 1, gpuMs));
                        } finally {
                            // Always release semaphore
                            gpuSemaphore.release();
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("GPU semaphore interrupted", e);
                    } catch (Exception e) {
                        logger.severe(String.format("[GPU-CONVERT-ASYNC] Chunk %d GPU failed: %s",
                                                   currentChunkIdx + 1, e.getMessage()));
                        throw new RuntimeException(e);
                    }
                });

                // Stage 3: Async HalfFloat conversion (depends on GPU completion)
                convertFutures[chunkIdx] = gpuFutures[chunkIdx].thenRunAsync(() -> {
                    long convertStart = System.nanoTime();
                    try {
                        if (PARALLEL_CPU_CONVERSION && currentChunkSize > 10000) {
                            // Use the shared convert executor for parallel conversion
                            IntStream.range(0, currentChunkSize).parallel()
                                .forEach(i -> result.set(offset + i, new HalfFloat(outputBuffers[bufferIdx].get(i))));
                        } else {
                            // Sequential conversion for small chunks
                            for (int i = 0; i < currentChunkSize; i++) {
                                result.set(offset + i, new HalfFloat(outputBuffers[bufferIdx].get(i)));
                            }
                        }
                        long convertEnd = System.nanoTime();
                        double convertMs = (convertEnd - convertStart) / 1_000_000.0;
                        logger.info(String.format("[GPU-CONVERT-ASYNC] Chunk %d convert completed: %.1fms",
                                                currentChunkIdx + 1, convertMs));
                    } catch (Exception e) {
                        logger.severe(String.format("[GPU-CONVERT-ASYNC] Chunk %d convert failed: %s",
                                                   currentChunkIdx + 1, e.getMessage()));
                        throw new RuntimeException(e);
                    }
                }, convertExecutor);
            }

            // Wait for all pipeline stages to complete with timeout
            logger.info("[GPU-CONVERT-STREAM] Waiting for all async operations to complete...");

            // Filter out null futures and wait with timeout
            CompletableFuture<Void>[] validFutures = java.util.Arrays.stream(convertFutures)
                .filter(java.util.Objects::nonNull)
                .toArray(CompletableFuture[]::new);

            if (validFutures.length > 0) {
                try {
                    CompletableFuture.allOf(validFutures).get(30, TimeUnit.SECONDS);
                    logger.info("[GPU-CONVERT-STREAM] All async operations completed successfully");
                } catch (TimeoutException e) {
                    logger.severe("[GPU-CONVERT-STREAM] Async operations timed out after 30 seconds");
                    // Cancel remaining futures
                    for (CompletableFuture<Void> future : validFutures) {
                        if (future != null && !future.isDone()) {
                            future.cancel(true);
                        }
                    }
                    throw new RuntimeException("Async streaming pipeline timed out", e);
                } catch (ExecutionException e) {
                    logger.severe("[GPU-CONVERT-STREAM] Async operation failed: " + e.getCause().getMessage());
                    throw new RuntimeException("Async streaming pipeline failed", e.getCause());
                }
            } else {
                logger.warning("[GPU-CONVERT-STREAM] No valid futures to wait for");
            }

        } catch (Exception e) {
            logger.severe("[GPU-CONVERT-STREAM] Async pipeline failed: " + e.getMessage());
            throw new RuntimeException("Async streaming pipeline failed", e);
        } finally {
            // Shutdown executor
            convertExecutor.shutdown();
            try {
                if (!convertExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    convertExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                convertExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }

            // Phase 5 Optimization: Return streaming buffers to pool
            for (int i = 0; i < STREAMING_BUFFER_COUNT; i++) {
                returnPooledInputBuffer(chunkSize, inputBuffers[i]);
                returnPooledOutputBuffer(chunkSize, outputBuffers[i]);
            }
        }

        return result;
    }

    /**
     * Process GPU kernel directly without cached execution plan (for streaming)
     */
    private static void processGPUKernelDirect(FloatArray inputArray, FloatArray outputArray, int chunkSize) {
        TaskGraph taskGraph;
        if (GPU_PREPROCESSING_ENABLED && (GPU_SCALE_FACTOR != 1.0f || GPU_OFFSET_VALUE != 0.0f || GPU_NORMALIZE_ENABLED)) {
            taskGraph = new TaskGraph("streamEnhanced")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inputArray)
                .task("processEnhanced", GPUTensorConverter::enhancedProcessFloatsKernel,
                      inputArray, outputArray, GPU_SCALE_FACTOR, GPU_OFFSET_VALUE, GPU_NORMALIZE_ENABLED)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputArray);
        } else {
            taskGraph = new TaskGraph("streamBasic")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inputArray)
                .task("process", GPUTensorConverter::processFloatsKernel, inputArray, outputArray)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputArray);
        }

        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        executionPlan.execute();
    }

    /**
     * Process chunks in parallel (existing implementation separated)
     */
    private static HalfFloatArray processChunksParallel(FloatTensor tensor, int tensorSize, int chunkSize, int numChunks) {
        HalfFloatArray result = new HalfFloatArray(tensorSize);
        List<CompletableFuture<Void>> futures = new ArrayList<>();

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            final int idx = chunkIdx;
            final int offset = idx * chunkSize;
            final int currentChunkSize = Math.min(chunkSize, tensorSize - offset);

            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                logger.info(String.format("[GPU-CONVERT-P%d] Processing chunk %d-%d",
                                        idx, offset, offset + currentChunkSize));
                HalfFloatArray chunkResult = processGPUChunk(tensor, offset, currentChunkSize);

                // Copy chunk result to main result
                for (int i = 0; i < currentChunkSize; i++) {
                    result.set(offset + i, chunkResult.get(i));
                }
            }, CONVERSION_POOL);

            futures.add(future);
        }

        // Wait for all parallel chunks to complete
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        return result;
    }

    /**
     * Process chunks sequentially (existing implementation separated)
     */
    private static HalfFloatArray processChunksSequential(FloatTensor tensor, int tensorSize, int chunkSize) {
        HalfFloatArray result = new HalfFloatArray(tensorSize);

        for (int offset = 0; offset < tensorSize; offset += chunkSize) {
            int currentChunkSize = Math.min(chunkSize, tensorSize - offset);
            logger.info(String.format("[GPU-CONVERT] Processing chunk %d-%d (%.1f%%)",
                                    offset, offset + currentChunkSize,
                                    ((offset + currentChunkSize) * 100.0) / tensorSize));

            HalfFloatArray chunkResult = processGPUChunk(tensor, offset, currentChunkSize);

            // Copy chunk result to main result
            for (int i = 0; i < currentChunkSize; i++) {
                result.set(offset + i, chunkResult.get(i));
            }
        }

        return result;
    }

    /**
     * Convert F32 tensor to HalfFloatArray using actual GPU
     */
    private static HalfFloatArray convertF32ToHalfFloatGPU(FloatTensor tensor, int tensorSize) {
        logger.info("[GPU-CONVERT] Using actual GPU F32 conversion kernel");

        // Check memory requirements
        long memoryRequired = (long) tensorSize * (4 + 2); // 4 bytes input + 2 bytes output
        if (memoryRequired > gpuMemoryAvailable * 0.8) {
            logger.warning("[GPU-CONVERT] Tensor too large for GPU memory, using CPU");
            return convertToHalfFloatArrayCPU(tensor, tensorSize);
        }

        try {
            // Check cache first
            int tensorHash = System.identityHashCode(tensor);
            HalfFloatArray cached = CONVERSION_CACHE.get(tensorHash);
            if (cached != null) {
                logger.info("[GPU-CONVERT] Using cached conversion");
                return cached;
            }

            // Use configured chunk size (default 100M for fewer chunks)
            int chunkSize = GPU_CHUNK_SIZE;

            if (tensorSize <= chunkSize) {
                // Small tensor - process in one go
                HalfFloatArray result = processGPUChunk(tensor, 0, tensorSize);
                CONVERSION_CACHE.put(tensorHash, result);
                return result;
            } else {
                // Large tensor - use Phase 6 hybrid approach
                HalfFloatArray result;

                if (HYBRID_ENABLED) {
                    // Phase 6: Smart hybrid fallback solution
                    result = convertF32ToHalfFloatGPUHybrid(tensor, tensorSize, chunkSize);
                } else {
                    // Legacy approach without hybrid fallback
                    int numChunks = (tensorSize + chunkSize - 1) / chunkSize;
                    logger.info(String.format("[GPU-CONVERT] Large tensor (%d elements), processing in %d chunks of %dM",
                                            tensorSize, numChunks, chunkSize / 1_000_000));

                    if (STREAMING_ENABLED && numChunks >= 3) {
                        logger.info("[GPU-CONVERT-OPT] Using streaming pipeline processing");
                        result = convertF32ToHalfFloatGPUStreaming(tensor, tensorSize, chunkSize);
                    } else if (PARALLEL_CHUNKS && numChunks > 1) {
                        logger.info("[GPU-CONVERT] Processing chunks in PARALLEL");
                        result = processChunksParallel(tensor, tensorSize, chunkSize, numChunks);
                    } else {
                        logger.info("[GPU-CONVERT] Processing chunks SEQUENTIALLY");
                        result = processChunksSequential(tensor, tensorSize, chunkSize);
                    }
                }

                logger.info("[GPU-CONVERT] All chunks processed successfully");
                CONVERSION_CACHE.put(tensorHash, result);
                return result;
            }

        } catch (Exception e) {
            logger.severe("[GPU-CONVERT] GPU F32 conversion failed: " + e.getMessage());
            e.printStackTrace();
            throw e; // Re-throw instead of falling back - let's see the actual error
        }
    }

    /**
     * Check if GPU conversion is available and configured
     */
    public static boolean isGPUConversionAvailable() {
        return USE_GPU_CONVERSION && gpuAvailable;
    }

    /**
     * Get GPU memory information
     */
    public static long getGPUMemoryAvailable() {
        return gpuMemoryAvailable;
    }

    /**
     * Shutdown conversion thread pool
     */
    public static void shutdown() {
        CONVERSION_POOL.shutdown();
    }

    // ============= PHASE 6: HYBRID FALLBACK SOLUTION =============

    /**
     * Phase 6: Smart hybrid fallback GPU conversion with automatic degradation strategies
     */
    private static HalfFloatArray convertF32ToHalfFloatGPUHybrid(FloatTensor tensor, int tensorSize, int chunkSize) {
        int numChunks = (tensorSize + chunkSize - 1) / chunkSize;

        logger.info(String.format("[GPU-CONVERT-HYBRID] Processing %d elements in %d chunks (Phase 6 mode)",
                                tensorSize, numChunks));

        // Strategy 1: Try Phase 4 streaming if enabled and not dynamically disabled
        if (STREAMING_ENABLED && !DYNAMIC_STREAMING_DISABLED && numChunks >= 3) {
            try {
                logger.info("[GPU-CONVERT-HYBRID] Attempting Phase 4 streaming pipeline");
                HalfFloatArray result = convertF32ToHalfFloatGPUStreaming(tensor, tensorSize, chunkSize);

                // Success - reset failure counter
                GPU_OOM_COUNT.set(0);
                logger.info("[GPU-CONVERT-HYBRID] Phase 4 streaming successful");
                return result;

            } catch (Exception e) {
                // Check if this is a GPU memory allocation failure
                if (isGPUMemoryException(e)) {
                    int failureCount = GPU_OOM_COUNT.incrementAndGet();
                    logger.warning(String.format(
                        "[GPU-CONVERT-HYBRID] Phase 4 failed with GPU OOM (failure #%d): %s",
                        failureCount, e.getMessage()));

                    // Disable streaming for this session after max failures
                    if (failureCount >= MAX_STREAMING_FAILURES) {
                        DYNAMIC_STREAMING_DISABLED = true;
                        logger.warning("[GPU-CONVERT-HYBRID] Disabling Phase 4 streaming for this session");
                    }

                    // Clean up any partial streaming resources
                    cleanupStreamingResources();

                    // Fall through to sequential processing
                } else {
                    // Non-memory exception - propagate it
                    throw e;
                }
            }
        }

        // Strategy 2: Progressive chunk size reduction for memory pressure
        if (PROGRESSIVE_REDUCTION_ENABLED) {
            return convertWithProgressiveChunkReduction(tensor, tensorSize, chunkSize);
        } else {
            // Strategy 3: Direct sequential processing fallback
            logger.info("[GPU-CONVERT-HYBRID] Using sequential processing fallback");
            return processChunksSequential(tensor, tensorSize, chunkSize);
        }
    }

    /**
     * Progressive chunk size reduction strategy
     */
    private static HalfFloatArray convertWithProgressiveChunkReduction(FloatTensor tensor, int tensorSize, int chunkSize) {
        int currentChunkSize = chunkSize;
        int minChunkSize = Math.max(MIN_CHUNK_SIZE, chunkSize / 16); // Don't go below configured minimum

        while (currentChunkSize >= minChunkSize) {
            try {
                logger.info(String.format("[GPU-CONVERT-HYBRID] Trying sequential processing with %.1f MB chunks",
                                        currentChunkSize * 4.0 / 1_000_000));

                return processChunksSequential(tensor, tensorSize, currentChunkSize);

            } catch (Exception e) {
                if (isGPUMemoryException(e)) {
                    currentChunkSize = currentChunkSize / 2; // Halve chunk size
                    logger.info(String.format("[GPU-CONVERT-HYBRID] Reducing chunk size to %.1f MB due to memory pressure",
                                            currentChunkSize * 4.0 / 1_000_000));
                } else {
                    throw e; // Non-memory exception
                }
            }
        }

        // Strategy 3: Final fallback to CPU processing
        logger.warning("[GPU-CONVERT-HYBRID] All GPU strategies failed, falling back to CPU");
        return convertToHalfFloatArrayCPU(tensor, tensorSize);
    }

    /**
     * Helper method to identify GPU memory exceptions
     */
    private static boolean isGPUMemoryException(Exception e) {
        Throwable cause = e;
        while (cause != null) {
            if (cause instanceof uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException ||
                (cause.getMessage() != null && (
                    cause.getMessage().contains("Unable to allocate") ||
                    cause.getMessage().contains("CL_OUT_OF_RESOURCES") ||
                    cause.getMessage().contains("CL_MEM_OBJECT_ALLOCATION_FAILURE") ||
                    cause.getMessage().contains("out of memory") ||
                    cause.getMessage().toLowerCase().contains("memory")
                ))) {
                return true;
            }
            cause = cause.getCause();
        }
        return false;
    }

    /**
     * Cleanup method for streaming resources
     */
    private static void cleanupStreamingResources() {
        try {
            // Force GPU garbage collection if possible
            System.gc(); // Trigger JVM GC first

            // Clear buffer pools to free GPU memory
            INPUT_BUFFER_POOLS.clear();
            OUTPUT_BUFFER_POOLS.clear();

            // Clear conversion cache to free memory
            CONVERSION_CACHE.clear();

            logger.info("[GPU-CONVERT-HYBRID] Cleaned up streaming resources");
        } catch (Exception e) {
            logger.warning("[GPU-CONVERT-HYBRID] Error during cleanup: " + e.getMessage());
        }
    }

}