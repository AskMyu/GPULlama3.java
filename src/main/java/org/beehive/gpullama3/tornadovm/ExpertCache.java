package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * LRU Cache for OLMoE Expert Weights
 *
 * Manages dynamic loading and caching of expert weights for memory-constrained GPUs.
 * Uses LRU (Least Recently Used) eviction policy with frequency tracking.
 */
public class ExpertCache {
    private static final Logger logger = Logger.getLogger(ExpertCache.class.getName());

    // Expert weight structure
    public static class ExpertWeights {
        public FloatArray gateWeights;
        public FloatArray upWeights;
        public FloatArray downWeights;
        public int expertId;
        public long lastAccessTime;
        public int accessCount;

        public ExpertWeights(int expertId) {
            this.expertId = expertId;
            this.lastAccessTime = System.currentTimeMillis();
            this.accessCount = 0;
        }

        public long getMemorySize() {
            long size = 0;
            if (gateWeights != null) size += gateWeights.getSize() * 4L;
            if (upWeights != null) size += upWeights.getSize() * 4L;
            if (downWeights != null) size += downWeights.getSize() * 4L;
            return size;
        }
    }

    // Cache configuration
    private final int maxCapacity;
    private final Map<Integer, ExpertWeights> cache;
    private final LinkedList<Integer> lruQueue;
    private final Map<Integer, Integer> frequencyMap;

    // Cache statistics
    private long cacheHits = 0;
    private long cacheMisses = 0;
    private long totalLoads = 0;
    private long evictions = 0;

    // Weight loader callback (to be set by model loader)
    private ExpertWeightLoader weightLoader = null;

    /**
     * Interface for loading expert weights from storage
     */
    public interface ExpertWeightLoader {
        ExpertWeights loadExpertWeights(int expertId);
    }

    public ExpertCache(int maxCapacity) {
        this.maxCapacity = maxCapacity;
        this.cache = new ConcurrentHashMap<>();
        this.lruQueue = new LinkedList<>();
        this.frequencyMap = new ConcurrentHashMap<>();

        logger.info(String.format("[EXPERT-CACHE] Initialized with capacity: %d experts", maxCapacity));
    }

    /**
     * Set the weight loader callback
     */
    public void setWeightLoader(ExpertWeightLoader loader) {
        this.weightLoader = loader;
    }

    /**
     * Get expert weights, loading from storage if not cached
     */
    public synchronized ExpertWeights getExpertWeights(int expertId) {
        totalLoads++;

        // Check if already in cache
        if (cache.containsKey(expertId)) {
            // Cache hit
            cacheHits++;
            ExpertWeights weights = cache.get(expertId);
            weights.lastAccessTime = System.currentTimeMillis();
            weights.accessCount++;

            // Update LRU queue
            lruQueue.remove(Integer.valueOf(expertId));
            lruQueue.addFirst(expertId);

            // Update frequency
            frequencyMap.merge(expertId, 1, Integer::sum);

            return weights;
        }

        // Cache miss - need to load
        cacheMisses++;
        ExpertWeights weights = loadExpert(expertId);

        // Add to cache, evicting if necessary
        addToCache(expertId, weights);

        return weights;
    }

    /**
     * Get multiple expert weights efficiently
     */
    public synchronized ExpertWeights[] getExpertWeights(int[] expertIds) {
        ExpertWeights[] results = new ExpertWeights[expertIds.length];

        // Sort by cache presence to minimize loads
        List<Integer> toLoad = new ArrayList<>();
        for (int i = 0; i < expertIds.length; i++) {
            if (cache.containsKey(expertIds[i])) {
                results[i] = getExpertWeights(expertIds[i]);
            } else {
                toLoad.add(i);
            }
        }

        // Load missing experts
        for (int idx : toLoad) {
            results[idx] = getExpertWeights(expertIds[idx]);
        }

        return results;
    }

    /**
     * Load expert weights from storage
     */
    private ExpertWeights loadExpert(int expertId) {
        if (weightLoader == null) {
            logger.severe("[EXPERT-CACHE] No weight loader set!");
            throw new IllegalStateException("Expert weight loader not configured");
        }

        long startTime = System.currentTimeMillis();
        ExpertWeights weights = weightLoader.loadExpertWeights(expertId);
        long loadTime = System.currentTimeMillis() - startTime;

        logger.fine(String.format("[EXPERT-CACHE] Loaded expert %d in %d ms (%.2f MB)",
                                 expertId, loadTime, weights.getMemorySize() / (1024.0 * 1024.0)));

        return weights;
    }

    /**
     * Add expert to cache with eviction if necessary
     */
    private void addToCache(int expertId, ExpertWeights weights) {
        // Check if eviction needed
        while (cache.size() >= maxCapacity && !lruQueue.isEmpty()) {
            evictLRU();
        }

        // Add to cache
        cache.put(expertId, weights);
        lruQueue.addFirst(expertId);
        frequencyMap.merge(expertId, 1, Integer::sum);
    }

    /**
     * Evict least recently used expert
     */
    private void evictLRU() {
        if (lruQueue.isEmpty()) {
            return;
        }

        // Get LRU expert
        Integer expertToEvict = lruQueue.removeLast();

        // Consider frequency - don't evict frequently used experts
        int frequency = frequencyMap.getOrDefault(expertToEvict, 0);
        if (frequency > 10 && lruQueue.size() > 1) {
            // This expert is frequently used, try next LRU
            lruQueue.addFirst(expertToEvict); // Put it back at front
            expertToEvict = lruQueue.removeLast(); // Get next LRU
        }

        // Evict from cache
        ExpertWeights evicted = cache.remove(expertToEvict);
        if (evicted != null) {
            evictions++;
            logger.fine(String.format("[EXPERT-CACHE] Evicted expert %d (accessed %d times)",
                                     expertToEvict, evicted.accessCount));

            // Free GPU memory if needed
            freeExpertMemory(evicted);
        }

        // Reduce frequency count
        frequencyMap.compute(expertToEvict, (k, v) -> v == null || v <= 1 ? null : v - 1);
    }

    /**
     * Free GPU memory for evicted expert
     */
    private void freeExpertMemory(ExpertWeights weights) {
        // TornadoVM should handle this automatically via GC
        // But we null references to help GC
        weights.gateWeights = null;
        weights.upWeights = null;
        weights.downWeights = null;
    }

    /**
     * Pre-load experts based on prediction
     */
    public void preloadExperts(int[] expertIds) {
        logger.fine(String.format("[EXPERT-CACHE] Pre-loading %d experts", expertIds.length));

        for (int expertId : expertIds) {
            if (!cache.containsKey(expertId)) {
                ExpertWeights weights = loadExpert(expertId);
                addToCache(expertId, weights);
            }
        }
    }

    /**
     * Clear entire cache
     */
    public synchronized void clear() {
        for (ExpertWeights weights : cache.values()) {
            freeExpertMemory(weights);
        }
        cache.clear();
        lruQueue.clear();
        frequencyMap.clear();

        logger.info("[EXPERT-CACHE] Cache cleared");
    }

    /**
     * Get cache statistics
     */
    public void logCacheStatistics() {
        double hitRate = totalLoads > 0 ? (cacheHits * 100.0) / totalLoads : 0.0;
        long totalMemory = cache.values().stream()
            .mapToLong(ExpertWeights::getMemorySize)
            .sum();

        logger.info(String.format("[EXPERT-CACHE] Statistics: Hits=%d, Misses=%d, HitRate=%.1f%%, " +
                                 "Evictions=%d, CachedExperts=%d, Memory=%.2f MB",
                                 cacheHits, cacheMisses, hitRate, evictions,
                                 cache.size(), totalMemory / (1024.0 * 1024.0)));
    }

    /**
     * Get most frequently used experts (for prediction)
     */
    public List<Integer> getMostFrequentExperts(int count) {
        return frequencyMap.entrySet().stream()
            .sorted(Map.Entry.<Integer, Integer>comparingByValue().reversed())
            .limit(count)
            .map(Map.Entry::getKey)
            .toList();
    }

    // Getters for monitoring
    public int getCurrentSize() { return cache.size(); }
    public int getMaxCapacity() { return maxCapacity; }
    public double getCacheHitRate() {
        return totalLoads > 0 ? (cacheHits * 100.0) / totalLoads : 0.0;
    }
}