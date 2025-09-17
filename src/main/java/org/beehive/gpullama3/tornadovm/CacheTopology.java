package org.beehive.gpullama3.tornadovm;

import java.util.ArrayList;
import java.util.List;

/**
 * Cache topology analysis for optimizing batch allocation and access patterns.
 * 
 * This class analyzes the structure of transformer model caches to optimize
 * batch boundaries and memory access patterns for both VLM and non-VLM models.
 * 
 * Key responsibilities:
 * - Analyze layer structure and memory requirements
 * - Compute optimal batch boundaries aligned with layer boundaries
 * - Provide spatial locality optimization for different model types
 * - Support both standard language models and vision-language models
 */
public class CacheTopology {
    
    // Model configuration
    public final int layers;
    public final int contextLength;
    public final int kvDim;
    public final boolean isVLM;
    public final int visionTokens; // For VLM models, 0 for non-VLM
    
    // Computed properties
    public final long totalCacheSize;
    public final long layerCacheSize;
    public final LayerBatch[] optimalBatching;
    
    // Batch configuration
    private static final long BATCH_SIZE_FLOATS = 512L * 1024 * 1024 / 4; // 512MB in floats
    
    /**
     * Create topology for standard language models.
     */
    public CacheTopology(int layers, int contextLength, int kvDim) {
        this(layers, contextLength, kvDim, false, 0);
    }
    
    /**
     * Create topology for vision-language models.
     */
    public CacheTopology(int layers, int contextLength, int kvDim, boolean isVLM, int visionTokens) {
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvDim = kvDim;
        this.isVLM = isVLM;
        this.visionTokens = visionTokens;
        
        // Calculate cache sizes
        this.layerCacheSize = (long) contextLength * kvDim;
        this.totalCacheSize = layerCacheSize * layers;
        
        // Compute optimal batching strategy
        this.optimalBatching = computeOptimalBatching();
        
        System.out.printf("[CACHE-TOPOLOGY] %s model: %d layers, %d context, %d kvDim%n",
                        isVLM ? "VLM" : "LLM", layers, contextLength, kvDim);
        System.out.printf("[CACHE-TOPOLOGY] Cache: %.1f MB per layer, %.1f MB total%n",
                        (layerCacheSize * 4.0) / (1024 * 1024),
                        (totalCacheSize * 4.0) / (1024 * 1024));
        System.out.printf("[CACHE-TOPOLOGY] Optimal batching: %d batches%n", optimalBatching.length);
    }
    
    /**
     * Compute optimal batching strategy based on model architecture and access patterns.
     */
    private LayerBatch[] computeOptimalBatching() {
        List<LayerBatch> batches = new ArrayList<>();
        
        // Strategy: Align batch boundaries with layer boundaries when possible
        // This optimizes for the common access pattern where inference processes layers sequentially
        
        long currentBatchStart = 0;
        int currentLayerStart = 0;
        int batchIndex = 0;
        
        while (currentLayerStart < layers) {
            long remainingInBatch = BATCH_SIZE_FLOATS - (currentBatchStart % BATCH_SIZE_FLOATS);
            
            // How many complete layers can fit in the remaining batch space?
            int layersInBatch = (int) Math.min(remainingInBatch / layerCacheSize, layers - currentLayerStart);
            
            if (layersInBatch == 0) {
                // Layer doesn't fit - split layer across batches
                layersInBatch = 1;
            }
            
            int endLayer = currentLayerStart + layersInBatch - 1;
            long batchMemoryFootprint = layersInBatch * layerCacheSize * 4; // Convert to bytes
            
            LayerBatch batch = new LayerBatch(
                currentLayerStart, endLayer, batchIndex, 
                batchMemoryFootprint, computeAccessPattern(currentLayerStart, endLayer)
            );
            
            batches.add(batch);
            
            currentLayerStart += layersInBatch;
            currentBatchStart += layersInBatch * layerCacheSize;
            batchIndex++;
        }
        
        return batches.toArray(new LayerBatch[0]);
    }
    
    /**
     * Analyze access patterns for a range of layers.
     */
    private AccessPattern computeAccessPattern(int startLayer, int endLayer) {
        // For language models: Sequential layer access during inference
        // For VLM models: Additional consideration for vision token processing
        
        boolean hasSequentialAccess = true;
        boolean hasRandomAccess = isVLM && (startLayer == 0); // Vision processing often accesses early layers
        boolean hasCrossLayerDependency = (endLayer - startLayer) > 0;
        
        return new AccessPattern(hasSequentialAccess, hasRandomAccess, hasCrossLayerDependency);
    }
    
    /**
     * Get batch index for a specific layer.
     */
    public int getBatchForLayer(int layer) {
        for (LayerBatch batch : optimalBatching) {
            if (layer >= batch.startLayer && layer <= batch.endLayer) {
                return batch.batchIndex;
            }
        }
        throw new IllegalArgumentException("Layer " + layer + " not found in topology");
    }
    
    /**
     * Get layer batch information for a specific batch index.
     */
    public LayerBatch getBatch(int batchIndex) {
        if (batchIndex >= optimalBatching.length) {
            throw new IllegalArgumentException("Batch index " + batchIndex + " exceeds batch count " + optimalBatching.length);
        }
        return optimalBatching[batchIndex];
    }
    
    /**
     * Calculate global cache index for a specific position within a layer.
     */
    public long calculateCacheIndex(int layer, int position, int dim) {
        if (layer >= layers || position >= contextLength || dim >= kvDim) {
            throw new IllegalArgumentException(String.format(
                "Invalid cache coordinates: layer=%d (max %d), pos=%d (max %d), dim=%d (max %d)",
                layer, layers-1, position, contextLength-1, dim, kvDim-1));
        }
        
        return (long) layer * layerCacheSize + (long) position * kvDim + dim;
    }
    
    /**
     * Check if the cache requires batching (>2GB).
     */
    public boolean requiresBatching() {
        return totalCacheSize * 4L > Integer.MAX_VALUE - 1024L;
    }
    
    /**
     * Get memory efficiency statistics.
     */
    public MemoryEfficiencyStats getMemoryEfficiency() {
        long totalBatchMemory = 0;
        int crossBatchLayers = 0;
        
        for (LayerBatch batch : optimalBatching) {
            totalBatchMemory += batch.memoryFootprint;
            if (batch.startLayer != batch.endLayer) {
                crossBatchLayers++;
            }
        }
        
        double efficiency = (double) (totalCacheSize * 4) / totalBatchMemory;
        double layerAlignment = 1.0 - ((double) crossBatchLayers / optimalBatching.length);
        
        return new MemoryEfficiencyStats(efficiency, layerAlignment, optimalBatching.length);
    }
    
    /**
     * Print comprehensive topology analysis.
     */
    public void printAnalysis() {
        System.out.println("[CACHE-TOPOLOGY] === Topology Analysis ===");
        System.out.printf("Model type: %s%n", isVLM ? "Vision-Language Model" : "Language Model");
        if (isVLM) {
            System.out.printf("Vision tokens: %d%n", visionTokens);
        }
        
        System.out.printf("Layers: %d, Context: %d, KV Dimension: %d%n", layers, contextLength, kvDim);
        System.out.printf("Total cache size: %.1f MB%n", (totalCacheSize * 4.0) / (1024 * 1024));
        System.out.printf("Requires batching: %s%n", requiresBatching() ? "YES" : "NO");
        
        if (requiresBatching()) {
            System.out.printf("Batch strategy: %d batches%n", optimalBatching.length);
            
            for (int i = 0; i < optimalBatching.length; i++) {
                LayerBatch batch = optimalBatching[i];
                System.out.printf("  Batch %d: layers %d-%d, %.1f MB%n",
                                i, batch.startLayer, batch.endLayer,
                                batch.memoryFootprint / (1024.0 * 1024));
            }
            
            MemoryEfficiencyStats efficiency = getMemoryEfficiency();
            System.out.printf("Memory efficiency: %.2f%%, Layer alignment: %.2f%%%n",
                            efficiency.efficiency * 100, efficiency.layerAlignment * 100);
        }
    }
    
    // ================================================================================
    // NESTED CLASSES
    // ================================================================================
    
    /**
     * Represents a batch of layers with their memory and access characteristics.
     */
    public static class LayerBatch {
        public final int startLayer;
        public final int endLayer; 
        public final int batchIndex;
        public final long memoryFootprint; // In bytes
        public final AccessPattern accessPattern;
        
        public LayerBatch(int startLayer, int endLayer, int batchIndex, 
                         long memoryFootprint, AccessPattern accessPattern) {
            this.startLayer = startLayer;
            this.endLayer = endLayer;
            this.batchIndex = batchIndex;
            this.memoryFootprint = memoryFootprint;
            this.accessPattern = accessPattern;
        }
        
        public boolean containsLayer(int layer) {
            return layer >= startLayer && layer <= endLayer;
        }
        
        public int getLayerCount() {
            return endLayer - startLayer + 1;
        }
    }
    
    /**
     * Describes the access pattern characteristics for a batch.
     */
    public static class AccessPattern {
        public final boolean hasSequentialAccess;
        public final boolean hasRandomAccess;
        public final boolean hasCrossLayerDependency;
        
        public AccessPattern(boolean hasSequentialAccess, boolean hasRandomAccess, 
                           boolean hasCrossLayerDependency) {
            this.hasSequentialAccess = hasSequentialAccess;
            this.hasRandomAccess = hasRandomAccess;
            this.hasCrossLayerDependency = hasCrossLayerDependency;
        }
        
        public OptimizationStrategy getRecommendedStrategy() {
            if (hasSequentialAccess && !hasRandomAccess) {
                return OptimizationStrategy.SEQUENTIAL_PREFETCH;
            } else if (hasRandomAccess) {
                return OptimizationStrategy.KEEP_IN_MEMORY;
            } else if (hasCrossLayerDependency) {
                return OptimizationStrategy.CROSS_BATCH_COORDINATION;
            } else {
                return OptimizationStrategy.STANDARD;
            }
        }
    }
    
    /**
     * Optimization strategies for different access patterns.
     */
    public enum OptimizationStrategy {
        SEQUENTIAL_PREFETCH,    // Prefetch next batch when processing sequentially
        KEEP_IN_MEMORY,         // Keep frequently accessed batches in GPU memory
        CROSS_BATCH_COORDINATION, // Coordinate operations that span batches
        STANDARD                // Default batch processing
    }
    
    /**
     * Memory efficiency analysis results.
     */
    public static class MemoryEfficiencyStats {
        public final double efficiency;      // Ratio of used memory to allocated memory
        public final double layerAlignment;  // Fraction of layers that don't cross batch boundaries  
        public final int batchCount;
        
        public MemoryEfficiencyStats(double efficiency, double layerAlignment, int batchCount) {
            this.efficiency = efficiency;
            this.layerAlignment = layerAlignment;
            this.batchCount = batchCount;
        }
        
        public boolean isOptimal() {
            return efficiency > 0.95 && layerAlignment > 0.8;
        }
    }
}