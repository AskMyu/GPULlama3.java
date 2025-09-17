package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * CacheAllocationStrategy provides intelligent cache allocation that detects and handles
 * the TornadoVM 2GB FloatArray limit while maintaining backward compatibility.
 * 
 * Phase 1: Smart detection with clear error messages and fallback strategies
 * Phase 2: (Future) Full chunked array support with TornadoVM kernel updates
 * 
 * This implementation prioritizes stability and clear error reporting while providing
 * a foundation for future chunked cache support.
 */
public class CacheAllocationStrategy {
    
    private static final long MAX_SAFE_SIZE_BYTES = Integer.MAX_VALUE - 1024; // Leave safety margin
    private static final long MAX_SAFE_SIZE_FLOATS = MAX_SAFE_SIZE_BYTES / 4L;
    
    public static class AllocationResult {
        public final FloatArray array;
        public final boolean isLimited;
        public final String message;
        public final long requestedSize;
        public final long allocatedSize;
        
        public AllocationResult(FloatArray array, boolean isLimited, String message, 
                              long requestedSize, long allocatedSize) {
            this.array = array;
            this.isLimited = isLimited;
            this.message = message;
            this.requestedSize = requestedSize;
            this.allocatedSize = allocatedSize;
        }
    }
    
    /**
     * Attempts to allocate a cache array with intelligent size management.
     * 
     * @param requestedSize Number of floats requested
     * @param contextLength Original context length for fallback calculation
     * @param kvDim KV dimension for fallback calculation  
     * @param numLayers Number of layers for fallback calculation
     * @return AllocationResult with array and metadata
     */
    public static AllocationResult allocateCacheArray(long requestedSize, int contextLength, 
                                                    int kvDim, int numLayers) {
        
        long requestedBytes = requestedSize * 4L;
        
        System.out.println("CacheAllocationStrategy: Analyzing cache requirements:");
        System.out.println("  Requested: " + requestedSize + " floats (" + (requestedBytes / (1024*1024)) + "MB)");
        System.out.println("  TornadoVM limit: " + MAX_SAFE_SIZE_FLOATS + " floats (" + (MAX_SAFE_SIZE_BYTES / (1024*1024)) + "MB)");
        
        if (requestedSize <= MAX_SAFE_SIZE_FLOATS && requestedBytes <= MAX_SAFE_SIZE_BYTES) {
            // Standard allocation - fits within limits
            System.out.println("  ✓ Within limits - using standard FloatArray allocation");
            FloatArray array = new FloatArray((int) requestedSize);
            return new AllocationResult(array, false, 
                "Standard allocation successful", requestedSize, requestedSize);
        } else {
            // Exceeds limits - need fallback strategy
            System.out.println("  ⚠ Exceeds TornadoVM FloatArray 2GB limit");
            System.out.println("  Root cause: " + requestedBytes + " bytes > " + MAX_SAFE_SIZE_BYTES + " bytes");
            
            // Calculate reduced context length that fits within limits
            long maxFloatsPerLayer = MAX_SAFE_SIZE_FLOATS / numLayers;
            int maxContextLength = (int) (maxFloatsPerLayer / kvDim);
            int fallbackContextLength = Math.min(Math.max(maxContextLength - 64, 512), 2048); // Conservative with safety margin
            
            long fallbackCacheSize = (long) fallbackContextLength * kvDim * numLayers;
            
            System.out.println("  Phase 1 Solution: Using reduced context length for compatibility");
            System.out.println("  Original context: " + contextLength + " → Fallback context: " + fallbackContextLength);
            System.out.println("  Fallback cache: " + fallbackCacheSize + " floats (" + (fallbackCacheSize * 4 / (1024*1024)) + "MB)");
            
            FloatArray array = new FloatArray((int) fallbackCacheSize);
            
            String message = String.format(
                "Model requires %dMB cache but TornadoVM FloatArray limit is %dMB. " +
                "Using reduced context length (%d→%d) for compatibility. " +
                "This enables basic functionality while chunked cache architecture is in development.",
                requestedBytes / (1024*1024), MAX_SAFE_SIZE_BYTES / (1024*1024), 
                contextLength, fallbackContextLength);
                
            return new AllocationResult(array, true, message, requestedSize, fallbackCacheSize);
        }
    }
    
    /**
     * Check if allocation would exceed TornadoVM limits.
     */
    public static boolean wouldExceedLimits(long requestedSize) {
        long requestedBytes = requestedSize * 4L;
        return requestedSize > MAX_SAFE_SIZE_FLOATS || requestedBytes > MAX_SAFE_SIZE_BYTES;
    }
    
    /**
     * Calculate maximum safe context length for given model parameters.
     */
    public static int calculateMaxSafeContextLength(int kvDim, int numLayers) {
        long maxFloatsPerLayer = MAX_SAFE_SIZE_FLOATS / numLayers;
        return Math.max((int) (maxFloatsPerLayer / kvDim) - 64, 512); // Conservative with safety margin
    }
    
    /**
     * Get detailed analysis of cache requirements vs limits.
     */
    public static String analyzeRequirements(int contextLength, int kvDim, int numLayers) {
        long totalFloats = (long) contextLength * kvDim * numLayers;
        long totalBytes = totalFloats * 4L;
        
        StringBuilder analysis = new StringBuilder();
        analysis.append("Cache Requirement Analysis:\n");
        analysis.append(String.format("  Model: context=%d, kvDim=%d, layers=%d\n", contextLength, kvDim, numLayers));
        analysis.append(String.format("  Required cache: %d floats (%.1fMB)\n", totalFloats, totalBytes / (1024.0*1024.0)));
        analysis.append(String.format("  TornadoVM limit: %d floats (%.1fMB)\n", MAX_SAFE_SIZE_FLOATS, MAX_SAFE_SIZE_BYTES / (1024.0*1024.0)));
        
        if (wouldExceedLimits(totalFloats)) {
            analysis.append("  Status: ⚠ EXCEEDS LIMITS\n");
            int maxSafeContext = calculateMaxSafeContextLength(kvDim, numLayers);
            analysis.append(String.format("  Max safe context: %d (%.1f%% of requested)\n", 
                maxSafeContext, (double) maxSafeContext / contextLength * 100.0));
        } else {
            analysis.append("  Status: ✓ Within limits\n");
        }
        
        return analysis.toString();
    }
}