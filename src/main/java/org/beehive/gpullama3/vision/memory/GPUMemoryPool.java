package org.beehive.gpullama3.vision.memory;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.Map;
import java.util.Queue;

/**
 * Centralized GPU Memory Pool for VLM operations.
 * 
 * Manages VRAM allocation, tracking, and cleanup for Vision Language Models.
 * Prevents memory fragmentation and provides emergency cleanup when approaching limits.
 * 
 * Features:
 * - Centralized VRAM usage tracking across all VLM components
 * - Buffer pool for reusing common sizes (patch buffers, embedding buffers)  
 * - Emergency cleanup when approaching configured memory limits
 * - Thread-safe operations for concurrent VLM inference
 * - Detailed memory usage statistics and monitoring
 * 
 * Based on 2025 research: vLLM PagedAttention and NVIDIA NIM memory management.
 */
public class GPUMemoryPool {
    
    // Memory configuration
    private static final long DEFAULT_MAX_VRAM_MB = 8 * 1024; // 8GB default
    private static final long EMERGENCY_THRESHOLD = 90; // 90% usage triggers cleanup
    private static final long WARNING_THRESHOLD = 75;   // 75% usage triggers warning
    
    // Buffer size categories for common VLM operations
    private static final int PATCH_BUFFER_SIZE = 576 * 14 * 14 * 3;     // 336x336 image patches
    private static final int EMBEDDING_BUFFER_SIZE = 576 * 1024;        // CLIP embeddings
    private static final int PROJECTION_BUFFER_SIZE = 576 * 4096;       // MLP projector output
    private static final int TRANSFORMER_BUFFER_SIZE = 577 * 1024;      // With class token
    
    // Singleton instance
    private static volatile GPUMemoryPool instance;
    private static final ReentrantLock instanceLock = new ReentrantLock();
    
    // Core memory tracking
    private final AtomicLong totalAllocatedBytes = new AtomicLong(0);
    private final AtomicLong peakUsageBytes = new AtomicLong(0);
    private final long maxVRAMBytes;
    
    // Buffer pools for common sizes
    private final Map<Integer, Queue<FloatArray>> bufferPools = new ConcurrentHashMap<>();
    private final Map<FloatArray, BufferInfo> activeBuffers = new ConcurrentHashMap<>();
    
    // Statistics
    private final AtomicLong allocationCount = new AtomicLong(0);
    private final AtomicLong reuseCount = new AtomicLong(0);
    private final AtomicLong emergencyCleanupCount = new AtomicLong(0);
    
    // Thread safety
    private final ReentrantLock allocationLock = new ReentrantLock();
    
    /**
     * Buffer metadata for tracking and cleanup.
     */
    private static class BufferInfo {
        final int sizeInFloats;
        final String allocatedFor;
        final long allocationTime;
        volatile boolean inUse;
        
        BufferInfo(int sizeInFloats, String allocatedFor) {
            this.sizeInFloats = sizeInFloats;
            this.allocatedFor = allocatedFor;
            this.allocationTime = System.currentTimeMillis();
            this.inUse = true;
        }
    }
    
    private GPUMemoryPool(long maxVRAMBytes) {
        this.maxVRAMBytes = maxVRAMBytes;
        
        // Initialize buffer pools for common sizes
        bufferPools.put(PATCH_BUFFER_SIZE, new ConcurrentLinkedQueue<>());
        bufferPools.put(EMBEDDING_BUFFER_SIZE, new ConcurrentLinkedQueue<>());
        bufferPools.put(PROJECTION_BUFFER_SIZE, new ConcurrentLinkedQueue<>());
        bufferPools.put(TRANSFORMER_BUFFER_SIZE, new ConcurrentLinkedQueue<>());
        
        System.err.printf("[GPU-MEMORY-POOL] Initialized with %d MB VRAM limit%n", maxVRAMBytes / (1024 * 1024));
        System.err.printf("[GPU-MEMORY-POOL] Buffer pools: patch=%d, embedding=%d, projection=%d, transformer=%d%n",
                         PATCH_BUFFER_SIZE, EMBEDDING_BUFFER_SIZE, PROJECTION_BUFFER_SIZE, TRANSFORMER_BUFFER_SIZE);
    }
    
    /**
     * Get singleton instance with default VRAM limit.
     */
    public static GPUMemoryPool getInstance() {
        return getInstance(DEFAULT_MAX_VRAM_MB * 1024 * 1024);
    }
    
    /**
     * Get singleton instance with specified VRAM limit.
     */
    public static GPUMemoryPool getInstance(long maxVRAMBytes) {
        if (instance == null) {
            instanceLock.lock();
            try {
                if (instance == null) {
                    instance = new GPUMemoryPool(maxVRAMBytes);
                }
            } finally {
                instanceLock.unlock();
            }
        }
        return instance;
    }
    
    /**
     * Allocate GPU buffer for VLM operations with automatic pool management.
     * 
     * @param sizeInFloats Size in number of floats
     * @param purpose Description of what this buffer is for
     * @return FloatArray buffer, either reused from pool or newly allocated
     */
    public FloatArray allocateVisionBuffer(int sizeInFloats, String purpose) {
        allocationLock.lock();
        try {
            // Check memory pressure before allocation
            long currentUsage = totalAllocatedBytes.get();
            long requestedBytes = (long) sizeInFloats * 4L;
            long afterAllocationUsage = currentUsage + requestedBytes;
            
            // Emergency cleanup if approaching limit
            if (afterAllocationUsage > maxVRAMBytes * EMERGENCY_THRESHOLD / 100) {
                System.err.printf("[GPU-MEMORY-POOL] EMERGENCY: Usage would be %d MB / %d MB (%.1f%%), triggering cleanup%n",
                                afterAllocationUsage / (1024 * 1024), maxVRAMBytes / (1024 * 1024),
                                100.0 * afterAllocationUsage / maxVRAMBytes);
                performEmergencyCleanup();
                emergencyCleanupCount.incrementAndGet();
            }
            
            // Try to reuse from buffer pool first
            Queue<FloatArray> pool = bufferPools.get(sizeInFloats);
            if (pool != null) {
                FloatArray reused = pool.poll();
                if (reused != null) {
                    reuseCount.incrementAndGet();
                    activeBuffers.put(reused, new BufferInfo(sizeInFloats, purpose));
                    System.err.printf("[GPU-MEMORY-POOL] Reused buffer: %d floats for %s (pool hit)%n", sizeInFloats, purpose);
                    return reused;
                }
            }
            
            // Allocate new buffer
            FloatArray buffer = new FloatArray(sizeInFloats);
            allocationCount.incrementAndGet();
            
            // Track allocation
            BufferInfo info = new BufferInfo(sizeInFloats, purpose);
            activeBuffers.put(buffer, info);
            
            // Update memory usage tracking
            totalAllocatedBytes.addAndGet(requestedBytes);
            peakUsageBytes.updateAndGet(current -> Math.max(current, totalAllocatedBytes.get()));
            
            // Log allocation with memory pressure information
            long usagePercent = 100 * totalAllocatedBytes.get() / maxVRAMBytes;
            String pressureLevel = usagePercent > WARNING_THRESHOLD ? "HIGH" : "NORMAL";
            System.err.printf("[GPU-MEMORY-POOL] Allocated: %d floats (%d MB) for %s - Total: %d MB (%.1f%%, %s pressure)%n",
                             sizeInFloats, requestedBytes / (1024 * 1024), purpose,
                             totalAllocatedBytes.get() / (1024 * 1024), (double)usagePercent, pressureLevel);
            
            return buffer;
            
        } finally {
            allocationLock.unlock();
        }
    }
    
    /**
     * Return buffer to pool for reuse.
     */
    public void releaseBuffer(FloatArray buffer) {
        if (buffer == null) return;
        
        BufferInfo info = activeBuffers.remove(buffer);
        if (info == null) {
            System.err.println("[GPU-MEMORY-POOL] WARNING: Attempted to release untracked buffer");
            return;
        }
        
        // Mark as no longer in use
        info.inUse = false;
        
        // Try to return to appropriate pool
        Queue<FloatArray> pool = bufferPools.get(info.sizeInFloats);
        if (pool != null && pool.size() < 10) { // Limit pool size to prevent memory bloat
            pool.offer(buffer);
            System.err.printf("[GPU-MEMORY-POOL] Returned to pool: %d floats for %s%n", info.sizeInFloats, info.allocatedFor);
        } else {
            // Not a pooled size or pool full - update memory tracking for actual deallocation
            long releasedBytes = (long) info.sizeInFloats * 4L;
            totalAllocatedBytes.addAndGet(-releasedBytes);
            System.err.printf("[GPU-MEMORY-POOL] Released: %d floats (%d MB) for %s%n", 
                             info.sizeInFloats, releasedBytes / (1024 * 1024), info.allocatedFor);
        }
    }
    
    /**
     * Emergency cleanup to free memory when approaching limits.
     */
    private void performEmergencyCleanup() {
        System.err.println("[GPU-MEMORY-POOL] Starting emergency cleanup...");
        
        int freedBuffers = 0;
        long freedBytes = 0;
        
        // Clear all buffer pools
        for (Queue<FloatArray> pool : bufferPools.values()) {
            while (!pool.isEmpty()) {
                FloatArray buffer = pool.poll();
                if (buffer != null) {
                    freedBuffers++;
                    freedBytes += buffer.getSize() * 4L;
                }
            }
        }
        
        // Update tracking after pool clearing
        totalAllocatedBytes.addAndGet(-freedBytes);
        
        System.err.printf("[GPU-MEMORY-POOL] Emergency cleanup freed %d buffers (%d MB)%n", 
                         freedBuffers, freedBytes / (1024 * 1024));
        
        // Force garbage collection as last resort
        System.gc();
        
        System.err.printf("[GPU-MEMORY-POOL] Post-cleanup usage: %d MB / %d MB (%.1f%%)%n",
                         totalAllocatedBytes.get() / (1024 * 1024), maxVRAMBytes / (1024 * 1024),
                         100.0 * totalAllocatedBytes.get() / maxVRAMBytes);
    }
    
    /**
     * Get current memory usage statistics.
     */
    public MemoryStats getStats() {
        return new MemoryStats(
            totalAllocatedBytes.get(),
            peakUsageBytes.get(),
            maxVRAMBytes,
            allocationCount.get(),
            reuseCount.get(),
            emergencyCleanupCount.get(),
            activeBuffers.size()
        );
    }
    
    /**
     * Memory statistics holder.
     */
    public static class MemoryStats {
        public final long currentUsageBytes;
        public final long peakUsageBytes;
        public final long maxVRAMBytes;
        public final long allocationCount;
        public final long reuseCount;
        public final long emergencyCleanupCount;
        public final int activeBufferCount;
        public final double currentUsagePercent;
        public final double peakUsagePercent;
        public final double reuseRate;
        
        MemoryStats(long currentUsageBytes, long peakUsageBytes, long maxVRAMBytes,
                   long allocationCount, long reuseCount, long emergencyCleanupCount, 
                   int activeBufferCount) {
            this.currentUsageBytes = currentUsageBytes;
            this.peakUsageBytes = peakUsageBytes;
            this.maxVRAMBytes = maxVRAMBytes;
            this.allocationCount = allocationCount;
            this.reuseCount = reuseCount;
            this.emergencyCleanupCount = emergencyCleanupCount;
            this.activeBufferCount = activeBufferCount;
            this.currentUsagePercent = 100.0 * currentUsageBytes / maxVRAMBytes;
            this.peakUsagePercent = 100.0 * peakUsageBytes / maxVRAMBytes;
            this.reuseRate = allocationCount > 0 ? 100.0 * reuseCount / (allocationCount + reuseCount) : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("GPU Memory: %d MB / %d MB (%.1f%%), Peak: %d MB (%.1f%%), " +
                               "Allocations: %d, Reuse: %d (%.1f%%), Active: %d, Cleanups: %d",
                               currentUsageBytes / (1024 * 1024), maxVRAMBytes / (1024 * 1024), currentUsagePercent,
                               peakUsageBytes / (1024 * 1024), peakUsagePercent,
                               allocationCount, reuseCount, reuseRate, activeBufferCount, emergencyCleanupCount);
        }
    }
    
    /**
     * Print current memory statistics.
     */
    public void printStats() {
        MemoryStats stats = getStats();
        System.err.println("[GPU-MEMORY-POOL] " + stats.toString());
    }
    
    /**
     * Shutdown and cleanup all resources.
     */
    public void shutdown() {
        allocationLock.lock();
        try {
            System.err.println("[GPU-MEMORY-POOL] Shutting down, final statistics:");
            printStats();
            
            // Clear all pools
            for (Queue<FloatArray> pool : bufferPools.values()) {
                pool.clear();
            }
            
            activeBuffers.clear();
            totalAllocatedBytes.set(0);
            
            System.err.println("[GPU-MEMORY-POOL] Shutdown complete");
        } finally {
            allocationLock.unlock();
        }
    }
}