package org.beehive.gpullama3.vision.cache;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import org.beehive.gpullama3.vision.memory.GPUMemoryPool;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import java.util.Map;

/**
 * Multi-Level Vision Cache for comprehensive VLM performance optimization.
 * 
 * Implements a hierarchical caching strategy that caches results at multiple processing stages:
 * 1. CLIP Vision Features (already exists in VisionFeatureCache)
 * 2. MLP Projector Results (most expensive operation - 1024D→4096D transformation)
 * 3. Patch Embeddings (intermediate results for reuse)
 * 4. Attention Weights (when transformer processing is optimized)
 * 
 * Based on 2025 research: NVIDIA NIM multi-tier caching and vLLM optimization strategies.
 */
public class MultiLevelVisionCache {
    
    // Cache configuration
    private static final int DEFAULT_CLIP_CACHE_SIZE = 50;
    private static final int DEFAULT_MLP_CACHE_SIZE = 100;      // MLP results are most valuable to cache
    private static final int DEFAULT_PATCH_CACHE_SIZE = 30;     // Intermediate patch embeddings
    private static final int DEFAULT_ATTENTION_CACHE_SIZE = 20; // Future: attention computation results
    
    // Individual cache layers
    private final VisionFeatureCache clipFeatureCache;
    private final Map<String, CachedMLPResult> mlpProjectionCache;
    private final Map<String, CachedPatchEmbedding> patchEmbeddingCache;
    private final Map<String, CachedAttentionResult> attentionCache;
    
    // Cache size limits
    private final int maxMLPCacheSize;
    private final int maxPatchCacheSize;
    private final int maxAttentionCacheSize;
    
    // Statistics tracking
    private final AtomicLong clipHits = new AtomicLong(0);
    private final AtomicLong mlpHits = new AtomicLong(0);
    private final AtomicLong patchHits = new AtomicLong(0);
    private final AtomicLong attentionHits = new AtomicLong(0);
    
    private final AtomicLong clipMisses = new AtomicLong(0);
    private final AtomicLong mlpMisses = new AtomicLong(0);
    private final AtomicLong patchMisses = new AtomicLong(0);
    private final AtomicLong attentionMisses = new AtomicLong(0);
    
    // Memory management integration
    private final GPUMemoryPool memoryPool;
    
    /**
     * Cached MLP projection result with metadata.
     */
    private static class CachedMLPResult {
        final FloatArray projectedFeatures;
        final long timestamp;
        final int originalTokenCount;
        final int projectedDimension;
        final boolean wasTokenReduced;
        
        CachedMLPResult(FloatArray features, int originalTokens, int projectedDim, boolean tokenReduced) {
            // Create a defensive copy to avoid external modifications
            this.projectedFeatures = new FloatArray(features.getSize());
            for (int i = 0; i < features.getSize(); i++) {
                this.projectedFeatures.set(i, features.get(i));
            }
            this.timestamp = System.currentTimeMillis();
            this.originalTokenCount = originalTokens;
            this.projectedDimension = projectedDim;
            this.wasTokenReduced = tokenReduced;
        }
        
        FloatArray getProjectedFeaturesCopy() {
            FloatArray copy = new FloatArray(projectedFeatures.getSize());
            for (int i = 0; i < projectedFeatures.getSize(); i++) {
                copy.set(i, projectedFeatures.get(i));
            }
            return copy;
        }
    }
    
    /**
     * Cached patch embedding result.
     */
    private static class CachedPatchEmbedding {
        final FloatArray patchEmbeddings;
        final long timestamp;
        final int numPatches;
        final int embeddingDim;
        
        CachedPatchEmbedding(FloatArray embeddings, int numPatches, int embeddingDim) {
            this.patchEmbeddings = new FloatArray(embeddings.getSize());
            for (int i = 0; i < embeddings.getSize(); i++) {
                this.patchEmbeddings.set(i, embeddings.get(i));
            }
            this.timestamp = System.currentTimeMillis();
            this.numPatches = numPatches;
            this.embeddingDim = embeddingDim;
        }
        
        FloatArray getEmbeddingsCopy() {
            FloatArray copy = new FloatArray(patchEmbeddings.getSize());
            for (int i = 0; i < patchEmbeddings.getSize(); i++) {
                copy.set(i, patchEmbeddings.get(i));
            }
            return copy;
        }
    }
    
    /**
     * Cached attention computation result (future enhancement).
     */
    private static class CachedAttentionResult {
        final FloatArray attentionOutput;
        final long timestamp;
        
        CachedAttentionResult(FloatArray output) {
            this.attentionOutput = new FloatArray(output.getSize());
            for (int i = 0; i < output.getSize(); i++) {
                this.attentionOutput.set(i, output.get(i));
            }
            this.timestamp = System.currentTimeMillis();
        }
        
        FloatArray getOutputCopy() {
            FloatArray copy = new FloatArray(attentionOutput.getSize());
            for (int i = 0; i < attentionOutput.getSize(); i++) {
                copy.set(i, attentionOutput.get(i));
            }
            return copy;
        }
    }
    
    public MultiLevelVisionCache() {
        this(DEFAULT_CLIP_CACHE_SIZE, DEFAULT_MLP_CACHE_SIZE, DEFAULT_PATCH_CACHE_SIZE, DEFAULT_ATTENTION_CACHE_SIZE);
    }
    
    public MultiLevelVisionCache(int clipCacheSize, int mlpCacheSize, int patchCacheSize, int attentionCacheSize) {
        this.clipFeatureCache = new VisionFeatureCache(clipCacheSize);
        this.mlpProjectionCache = new ConcurrentHashMap<>();
        this.patchEmbeddingCache = new ConcurrentHashMap<>();
        this.attentionCache = new ConcurrentHashMap<>();
        
        this.maxMLPCacheSize = mlpCacheSize;
        this.maxPatchCacheSize = patchCacheSize;
        this.maxAttentionCacheSize = attentionCacheSize;
        
        this.memoryPool = GPUMemoryPool.getInstance();
        
        System.err.printf("[MULTI-LEVEL-CACHE] Initialized: CLIP=%d, MLP=%d, Patch=%d, Attention=%d%n",
                         clipCacheSize, mlpCacheSize, patchCacheSize, attentionCacheSize);
    }
    
    /**
     * Get or compute CLIP vision features (delegates to existing VisionFeatureCache).
     */
    public FloatArray getOrComputeClipFeatures(byte[] imageData, Supplier<FloatArray> computeFeatures) {
        try {
            FloatArray result = clipFeatureCache.getOrCompute(imageData, computeFeatures);
            clipHits.incrementAndGet();
            return result;
        } catch (Exception e) {
            clipMisses.incrementAndGet();
            return computeFeatures.get();
        }
    }
    
    /**
     * Get or compute MLP projection results - highest value caching target.
     * 
     * @param visionFeaturesHash Hash of the input vision features
     * @param tokenCount Number of input tokens
     * @param inputDim Input dimension (1024 for CLIP)
     * @param outputDim Output dimension (4096 for LLaVA)
     * @param tokenReduced Whether token reduction was applied
     * @param computeProjection Supplier to compute projection if not cached
     * @return Projected features
     */
    public FloatArray getOrComputeMLPProjection(String visionFeaturesHash, int tokenCount, int inputDim, int outputDim,
                                               boolean tokenReduced, Supplier<FloatArray> computeProjection) {
        // Create cache key based on vision features and projection parameters
        String cacheKey = String.format("%s_mlp_%d_%d_%d_%b", visionFeaturesHash, tokenCount, inputDim, outputDim, tokenReduced);
        
        // Check cache first
        CachedMLPResult cached = mlpProjectionCache.get(cacheKey);
        if (cached != null) {
            mlpHits.incrementAndGet();
            System.err.printf("[MULTI-LEVEL-CACHE] MLP cache hit: %s (%d→%d, tokens=%d, reduced=%b)%n", 
                             visionFeaturesHash.substring(0, 8), inputDim, outputDim, tokenCount, tokenReduced);
            return cached.getProjectedFeaturesCopy();
        }
        
        // Cache miss - compute projection
        mlpMisses.incrementAndGet();
        System.err.printf("[MULTI-LEVEL-CACHE] MLP cache miss: %s - computing projection%n", 
                         visionFeaturesHash.substring(0, 8));
        
        FloatArray projected = computeProjection.get();
        
        // Add to cache with LRU eviction
        if (mlpProjectionCache.size() >= maxMLPCacheSize) {
            evictOldestMLP();
        }
        
        mlpProjectionCache.put(cacheKey, new CachedMLPResult(projected, tokenCount, outputDim, tokenReduced));
        
        return projected;
    }
    
    /**
     * Get or compute patch embeddings (intermediate caching).
     */
    public FloatArray getOrComputePatchEmbeddings(String patchHash, int numPatches, int embeddingDim,
                                                 Supplier<FloatArray> computeEmbeddings) {
        String cacheKey = String.format("%s_patch_%d_%d", patchHash, numPatches, embeddingDim);
        
        CachedPatchEmbedding cached = patchEmbeddingCache.get(cacheKey);
        if (cached != null) {
            patchHits.incrementAndGet();
            System.err.printf("[MULTI-LEVEL-CACHE] Patch embedding cache hit: %s%n", patchHash.substring(0, 8));
            return cached.getEmbeddingsCopy();
        }
        
        patchMisses.incrementAndGet();
        FloatArray embeddings = computeEmbeddings.get();
        
        if (patchEmbeddingCache.size() >= maxPatchCacheSize) {
            evictOldestPatch();
        }
        
        patchEmbeddingCache.put(cacheKey, new CachedPatchEmbedding(embeddings, numPatches, embeddingDim));
        
        return embeddings;
    }
    
    /**
     * Get or compute attention results (future enhancement for transformer optimization).
     */
    public FloatArray getOrComputeAttentionResult(String inputHash, Supplier<FloatArray> computeAttention) {
        CachedAttentionResult cached = attentionCache.get(inputHash);
        if (cached != null) {
            attentionHits.incrementAndGet();
            return cached.getOutputCopy();
        }
        
        attentionMisses.incrementAndGet();
        FloatArray result = computeAttention.get();
        
        if (attentionCache.size() >= maxAttentionCacheSize) {
            evictOldestAttention();
        }
        
        attentionCache.put(inputHash, new CachedAttentionResult(result));
        
        return result;
    }
    
    /**
     * Compute hash for FloatArray contents (for intermediate caching).
     */
    public String computeFloatArrayHash(FloatArray data, int samplePoints) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            
            // Sample key points in the array for efficient hashing
            int step = Math.max(1, data.getSize() / samplePoints);
            for (int i = 0; i < data.getSize(); i += step) {
                int bits = Float.floatToIntBits(data.get(i));
                digest.update((byte) (bits >> 24));
                digest.update((byte) (bits >> 16));
                digest.update((byte) (bits >> 8));
                digest.update((byte) bits);
            }
            
            byte[] hashBytes = digest.digest();
            StringBuilder hexString = new StringBuilder();
            for (byte b : hashBytes) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            return hexString.toString();
            
        } catch (NoSuchAlgorithmException e) {
            // Fallback to simple checksum
            long checksum = 0;
            for (int i = 0; i < data.getSize(); i++) {
                checksum += Float.floatToIntBits(data.get(i));
            }
            return Long.toHexString(checksum);
        }
    }
    
    /**
     * LRU eviction for MLP projection cache.
     */
    private void evictOldestMLP() {
        String oldestKey = null;
        long oldestTime = Long.MAX_VALUE;
        
        for (Map.Entry<String, CachedMLPResult> entry : mlpProjectionCache.entrySet()) {
            if (entry.getValue().timestamp < oldestTime) {
                oldestTime = entry.getValue().timestamp;
                oldestKey = entry.getKey();
            }
        }
        
        if (oldestKey != null) {
            mlpProjectionCache.remove(oldestKey);
            System.err.printf("[MULTI-LEVEL-CACHE] Evicted oldest MLP result: %s%n", oldestKey.substring(0, 16));
        }
    }
    
    /**
     * LRU eviction for patch embedding cache.
     */
    private void evictOldestPatch() {
        String oldestKey = null;
        long oldestTime = Long.MAX_VALUE;
        
        for (Map.Entry<String, CachedPatchEmbedding> entry : patchEmbeddingCache.entrySet()) {
            if (entry.getValue().timestamp < oldestTime) {
                oldestTime = entry.getValue().timestamp;
                oldestKey = entry.getKey();
            }
        }
        
        if (oldestKey != null) {
            patchEmbeddingCache.remove(oldestKey);
        }
    }
    
    /**
     * LRU eviction for attention cache.
     */
    private void evictOldestAttention() {
        String oldestKey = null;
        long oldestTime = Long.MAX_VALUE;
        
        for (Map.Entry<String, CachedAttentionResult> entry : attentionCache.entrySet()) {
            if (entry.getValue().timestamp < oldestTime) {
                oldestTime = entry.getValue().timestamp;
                oldestKey = entry.getKey();
            }
        }
        
        if (oldestKey != null) {
            attentionCache.remove(oldestKey);
        }
    }
    
    /**
     * Get comprehensive cache statistics across all levels.
     */
    public MultiLevelCacheStats getStats() {
        VisionFeatureCache.CacheStats clipStats = clipFeatureCache.getStats();
        
        return new MultiLevelCacheStats(
            clipStats.hits, clipStats.misses,
            mlpHits.get(), mlpMisses.get(),
            patchHits.get(), patchMisses.get(),
            attentionHits.get(), attentionMisses.get(),
            clipStats.currentSize, clipStats.maxSize,
            mlpProjectionCache.size(), maxMLPCacheSize,
            patchEmbeddingCache.size(), maxPatchCacheSize,
            attentionCache.size(), maxAttentionCacheSize
        );
    }
    
    /**
     * Multi-level cache statistics.
     */
    public static class MultiLevelCacheStats {
        public final long clipHits, clipMisses, mlpHits, mlpMisses, patchHits, patchMisses, attentionHits, attentionMisses;
        public final int clipSize, clipMax, mlpSize, mlpMax, patchSize, patchMax, attentionSize, attentionMax;
        public final double clipHitRate, mlpHitRate, patchHitRate, attentionHitRate, overallHitRate;
        
        MultiLevelCacheStats(long clipHits, long clipMisses, long mlpHits, long mlpMisses, 
                            long patchHits, long patchMisses, long attentionHits, long attentionMisses,
                            int clipSize, int clipMax, int mlpSize, int mlpMax, 
                            int patchSize, int patchMax, int attentionSize, int attentionMax) {
            this.clipHits = clipHits; this.clipMisses = clipMisses;
            this.mlpHits = mlpHits; this.mlpMisses = mlpMisses;
            this.patchHits = patchHits; this.patchMisses = patchMisses;
            this.attentionHits = attentionHits; this.attentionMisses = attentionMisses;
            
            this.clipSize = clipSize; this.clipMax = clipMax;
            this.mlpSize = mlpSize; this.mlpMax = mlpMax;
            this.patchSize = patchSize; this.patchMax = patchMax;
            this.attentionSize = attentionSize; this.attentionMax = attentionMax;
            
            this.clipHitRate = safeHitRate(clipHits, clipMisses);
            this.mlpHitRate = safeHitRate(mlpHits, mlpMisses);
            this.patchHitRate = safeHitRate(patchHits, patchMisses);
            this.attentionHitRate = safeHitRate(attentionHits, attentionMisses);
            
            long totalHits = clipHits + mlpHits + patchHits + attentionHits;
            long totalMisses = clipMisses + mlpMisses + patchMisses + attentionMisses;
            this.overallHitRate = safeHitRate(totalHits, totalMisses);
        }
        
        private double safeHitRate(long hits, long misses) {
            return (hits + misses > 0) ? (double) hits / (hits + misses) : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("Multi-Level Cache: Overall %.1f%% hit rate | " +
                               "CLIP: %.1f%% (%d/%d), MLP: %.1f%% (%d/%d), " +
                               "Patch: %.1f%% (%d/%d), Attention: %.1f%% (%d/%d)",
                               overallHitRate * 100,
                               clipHitRate * 100, clipSize, clipMax,
                               mlpHitRate * 100, mlpSize, mlpMax,
                               patchHitRate * 100, patchSize, patchMax,
                               attentionHitRate * 100, attentionSize, attentionMax);
        }
    }
    
    /**
     * Print comprehensive cache statistics.
     */
    public void printStats() {
        MultiLevelCacheStats stats = getStats();
        System.err.println("[MULTI-LEVEL-CACHE] " + stats.toString());
    }
    
    /**
     * Clear all cache levels.
     */
    public void clearAll() {
        clipFeatureCache.clear();
        mlpProjectionCache.clear();
        patchEmbeddingCache.clear();
        attentionCache.clear();
        
        // Reset statistics
        clipHits.set(0); clipMisses.set(0);
        mlpHits.set(0); mlpMisses.set(0);
        patchHits.set(0); patchMisses.set(0);
        attentionHits.set(0); attentionMisses.set(0);
        
        System.err.println("[MULTI-LEVEL-CACHE] All cache levels cleared");
    }
}