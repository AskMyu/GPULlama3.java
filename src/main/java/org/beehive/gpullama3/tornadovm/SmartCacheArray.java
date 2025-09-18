package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import org.beehive.gpullama3.tornadovm.OpenCLMemoryDetector;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Smart cache array that automatically handles large allocations through dynamic OpenCL-aware batch processing.
 *
 * This class automatically detects OpenCL device memory limits and splits large arrays into
 * manageable batches while maintaining a transparent API for existing code.
 *
 * Key features:
 * - Transparent API matching FloatArray
 * - Dynamic OpenCL memory limit detection
 * - Intelligent batching based on actual GPU constraints
 * - Zero performance impact for small arrays
 * - Full backward compatibility
 * - Automatic adaptation to different GPU configurations
 */
public class SmartCacheArray {
    
    // Dynamic OpenCL-aware memory limits (detected at runtime)
    private static volatile boolean limitsInitialized = false;
    private static volatile long dynamicMaxAllocationBytes;
    private static volatile long dynamicBatchSizeBytes;
    private static volatile int dynamicBatchSizeFloats;

    // Fallback limits for when OpenCL detection fails
    private static final long FALLBACK_MAX_ALLOCATION_BYTES = Integer.MAX_VALUE - 1024L; // 2GB safety
    private static final long FALLBACK_BATCH_SIZE_BYTES = 512L * 1024 * 1024; // 512MB batches
    
    // Core data structures
    private final FloatArray[] batches;
    private final int totalSize;
    private final int numBatches;
    private final boolean isBatched;
    private final BatchCacheManager batchManager;
    
    // Statistics and monitoring
    private final AtomicLong getOperations = new AtomicLong(0);
    private final AtomicLong setOperations = new AtomicLong(0);
    private final AtomicLong batchSwitches = new AtomicLong(0);
    private volatile long creationTime;
    
    /**
     * Create a SmartCacheArray with automatic size-based allocation strategy.
     * 
     * @param totalSize Total number of floats to allocate
     * @param topology Cache topology for optimization (can be null for simple allocation)
     */
    public SmartCacheArray(int totalSize, CacheTopology topology) {
        this.totalSize = totalSize;
        this.creationTime = System.nanoTime();
        
        // Initialize OpenCL-aware limits if needed
        initializeLimits();

        // Calculate total memory requirement
        long totalBytes = (long) totalSize * 4L + 24L; // 24 bytes TornadoVM header

        if (totalBytes <= dynamicMaxAllocationBytes) {
            // Standard allocation for small arrays
            this.batches = new FloatArray[]{new FloatArray(totalSize)};
            this.numBatches = 1;
            this.isBatched = false;
            this.batchManager = null;
            
            System.out.printf("[SMART-CACHE] Standard allocation: %d floats (%.1f MB)%n", 
                            totalSize, totalBytes / (1024.0 * 1024.0));
        } else {
            // Batch allocation for large arrays using dynamic batch size
            this.numBatches = (totalSize + dynamicBatchSizeFloats - 1) / dynamicBatchSizeFloats;
            this.batches = new FloatArray[numBatches];
            this.isBatched = true;

            System.out.printf("[SMART-CACHE] Large allocation detected: %.1f MB > %.1f MB limit%n",
                            totalBytes / (1024.0 * 1024.0),
                            dynamicMaxAllocationBytes / (1024.0 * 1024.0));

            // Create batches using OpenCL-aware batch size
            for (int i = 0; i < numBatches; i++) {
                int batchSize = Math.min(dynamicBatchSizeFloats, totalSize - i * dynamicBatchSizeFloats);
                this.batches[i] = new FloatArray(batchSize);
            }
            
            // Initialize batch manager if topology provided
            this.batchManager = (topology != null) ? 
                new BatchCacheManager(topology, batches) : null;
            
            System.out.printf("[SMART-CACHE] Batched allocation: %d floats (%.1f MB) in %d batches of %.1f MB each%n",
                            totalSize, totalBytes / (1024.0 * 1024.0), numBatches,
                            dynamicBatchSizeBytes / (1024.0 * 1024.0));
        }
    }
    
    /**
     * Create a SmartCacheArray with simple allocation (no topology optimization).
     */
    public SmartCacheArray(int totalSize) {
        this(totalSize, null);
    }

    /**
     * Initialize OpenCL memory limits dynamically.
     * Called once per application session.
     */
    private static synchronized void initializeLimits() {
        if (limitsInitialized) {
            return;
        }

        try {
            // Detect OpenCL capabilities
            long maxAlloc = OpenCLMemoryDetector.getMaxAllocationSize();
            long batchSize = OpenCLMemoryDetector.getRecommendedBatchSize();

            // Apply detected limits
            dynamicMaxAllocationBytes = maxAlloc;
            dynamicBatchSizeBytes = batchSize;
            dynamicBatchSizeFloats = (int)(batchSize / 4);

            System.out.printf("[SMART-CACHE] OpenCL limits detected: Max=%.2f GB, Batch=%.1f MB%n",
                            maxAlloc / (1024.0 * 1024.0 * 1024.0),
                            batchSize / (1024.0 * 1024.0));

        } catch (Exception e) {
            // Fallback to safe defaults
            dynamicMaxAllocationBytes = FALLBACK_MAX_ALLOCATION_BYTES;
            dynamicBatchSizeBytes = FALLBACK_BATCH_SIZE_BYTES;
            dynamicBatchSizeFloats = (int)(FALLBACK_BATCH_SIZE_BYTES / 4);

            System.out.printf("[SMART-CACHE] OpenCL detection failed, using fallbacks: Max=%.2f GB, Batch=%.1f MB%n",
                            FALLBACK_MAX_ALLOCATION_BYTES / (1024.0 * 1024.0 * 1024.0),
                            FALLBACK_BATCH_SIZE_BYTES / (1024.0 * 1024.0));
        }

        limitsInitialized = true;
    }
    
    // ================================================================================
    // TRANSPARENT API - Drop-in replacement for FloatArray
    // ================================================================================
    
    /**
     * Get value at specified index.
     * Transparent access across batches for large arrays.
     */
    public float get(int index) {
        getOperations.incrementAndGet();
        
        if (!isBatched) {
            return batches[0].get(index);
        }
        
        // Calculate batch and local index
        int batchIndex = index / dynamicBatchSizeFloats;
        int localIndex = index % dynamicBatchSizeFloats;
        
        if (batchIndex >= numBatches) {
            throw new IndexOutOfBoundsException("Index " + index + " exceeds array size " + totalSize);
        }
        
        batchSwitches.incrementAndGet();
        return batches[batchIndex].get(localIndex);
    }
    
    /**
     * Set value at specified index.
     * Transparent access across batches for large arrays.
     */
    public void set(int index, float value) {
        setOperations.incrementAndGet();
        
        if (!isBatched) {
            batches[0].set(index, value);
            return;
        }
        
        // Calculate batch and local index
        int batchIndex = index / dynamicBatchSizeFloats;
        int localIndex = index % dynamicBatchSizeFloats;
        
        if (batchIndex >= numBatches) {
            throw new IndexOutOfBoundsException("Index " + index + " exceeds array size " + totalSize);
        }
        
        batchSwitches.incrementAndGet();
        batches[batchIndex].set(localIndex, value);
    }
    
    /**
     * Get total size of the array.
     */
    public int getSize() {
        return totalSize;
    }
    
    // ================================================================================
    // BATCH-AWARE API - For advanced kernel integration
    // ================================================================================
    
    /**
     * Check if this array uses batch processing.
     */
    public boolean isBatched() {
        return isBatched;
    }
    
    /**
     * Get number of batches (1 for non-batched arrays).
     */
    public int getNumBatches() {
        return numBatches;
    }
    
    /**
     * Get specific batch array.
     */
    public FloatArray getBatch(int batchIndex) {
        if (batchIndex >= numBatches) {
            throw new IndexOutOfBoundsException("Batch index " + batchIndex + " exceeds batch count " + numBatches);
        }
        return batches[batchIndex];
    }
    
    /**
     * Get all batches for advanced processing.
     */
    public FloatArray[] getBatches() {
        return batches.clone(); // Defensive copy
    }
    
    /**
     * Get the size of a standard batch (except possibly the last one).
     */
    public int getBatchSize() {
        initializeLimits();
        return dynamicBatchSizeFloats;
    }
    
    /**
     * For non-batched arrays, get the direct FloatArray.
     * Throws exception if called on batched array.
     */
    public FloatArray getDirectArray() {
        if (isBatched) {
            throw new IllegalStateException("Cannot get direct array from batched SmartCacheArray");
        }
        return batches[0];
    }
    
    // ================================================================================
    // BATCH KERNEL EXECUTION
    // ================================================================================
    
    /**
     * Execute a TornadoVM kernel with batch coordination.
     * Only available if BatchCacheManager was initialized with topology.
     */
    public void executeBatchKernel(String kernelName, Object... args) {
        if (batchManager == null) {
            throw new IllegalStateException("Batch kernel execution requires CacheTopology during construction");
        }
        batchManager.executeBatched(kernelName, args);
    }
    
    // ================================================================================
    // UTILITY METHODS
    // ================================================================================
    
    /**
     * Calculate which batch contains the specified global index.
     */
    public int getBatchForIndex(int globalIndex) {
        if (globalIndex >= totalSize || globalIndex < 0) {
            throw new IndexOutOfBoundsException("Index " + globalIndex + " out of range [0, " + totalSize + ")");
        }
        return isBatched ? (globalIndex / dynamicBatchSizeFloats) : 0;
    }
    
    /**
     * Calculate local index within a batch for the specified global index.
     */
    public int getLocalIndex(int globalIndex) {
        if (globalIndex >= totalSize || globalIndex < 0) {
            throw new IndexOutOfBoundsException("Index " + globalIndex + " out of range [0, " + totalSize + ")");
        }
        return isBatched ? (globalIndex % dynamicBatchSizeFloats) : globalIndex;
    }
    
    /**
     * Copy data from another SmartCacheArray (optimized for batch operations).
     */
    public void copyFrom(SmartCacheArray source, int srcOffset, int destOffset, int length) {
        // Optimized batch-aware copying
        for (int i = 0; i < length; i++) {
            this.set(destOffset + i, source.get(srcOffset + i));
        }
    }
    
    /**
     * Copy data to standard FloatArray (for kernel interfacing).
     */
    public void copyTo(FloatArray dest, int srcOffset, int destOffset, int length) {
        for (int i = 0; i < length; i++) {
            dest.set(destOffset + i, this.get(srcOffset + i));
        }
    }
    
    // ================================================================================
    // MONITORING AND DIAGNOSTICS
    // ================================================================================
    
    /**
     * Get performance statistics for this cache array.
     */
    public CacheStatistics getStatistics() {
        long ageMs = (System.nanoTime() - creationTime) / 1_000_000;
        return new CacheStatistics(
            totalSize, numBatches, isBatched,
            getOperations.get(), setOperations.get(), batchSwitches.get(),
            ageMs
        );
    }
    
    /**
     * Print comprehensive diagnostics.
     */
    public void printDiagnostics() {
        CacheStatistics stats = getStatistics();
        System.out.printf("[SMART-CACHE] Diagnostics for %s cache:%n", isBatched ? "BATCHED" : "STANDARD");
        System.out.printf("  Size: %d floats (%.1f MB)%n", 
                        totalSize, (totalSize * 4.0) / (1024 * 1024));
        System.out.printf("  Batches: %d%n", numBatches);
        System.out.printf("  Operations: %d gets, %d sets, %d batch switches%n",
                        stats.getOperations, stats.setOperations, stats.batchSwitches);
        System.out.printf("  Age: %d ms%n", stats.ageMs);
        
        if (isBatched) {
            System.out.printf("  Batch efficiency: %.2f ops/switch%n",
                            (stats.getOperations + stats.setOperations) / (double)Math.max(1, stats.batchSwitches));
        }
    }
    
    /**
     * Statistics container class.
     */
    public static class CacheStatistics {
        public final int totalSize;
        public final int numBatches;
        public final boolean isBatched;
        public final long getOperations;
        public final long setOperations;
        public final long batchSwitches;
        public final long ageMs;
        
        public CacheStatistics(int totalSize, int numBatches, boolean isBatched,
                             long getOperations, long setOperations, long batchSwitches, long ageMs) {
            this.totalSize = totalSize;
            this.numBatches = numBatches;
            this.isBatched = isBatched;
            this.getOperations = getOperations;
            this.setOperations = setOperations;
            this.batchSwitches = batchSwitches;
            this.ageMs = ageMs;
        }
    }
}