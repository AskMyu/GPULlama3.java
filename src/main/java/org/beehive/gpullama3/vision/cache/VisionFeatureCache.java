package org.beehive.gpullama3.vision.cache;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Vision Feature Cache for caching computed vision features to avoid recomputation.
 * Implements LRU eviction policy and provides thread-safe access.
 * 
 * Performance benefits:
 * - Avoids expensive CLIP encoding recomputation for repeated images
 * - Reduces GPU/CPU usage for common images (logos, diagrams, etc.)
 * - Particularly beneficial during development and testing phases
 */
public class VisionFeatureCache {
    private final Map<String, CachedFeature> cache;
    private final int maxCacheSize;
    private final AtomicLong hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);
    
    /**
     * Cached feature entry with metadata.
     */
    private static class CachedFeature {
        final FloatArray features;
        final long timestamp;
        final int featureSize;
        
        CachedFeature(FloatArray features) {
            // Deep copy to avoid external modifications
            this.features = new FloatArray(features.getSize());
            for (int i = 0; i < features.getSize(); i++) {
                this.features.set(i, features.get(i));
            }
            this.timestamp = System.currentTimeMillis();
            this.featureSize = features.getSize();
        }
        
        FloatArray getFeaturesCopy() {
            // Return a copy to avoid external modifications
            FloatArray copy = new FloatArray(features.getSize());
            for (int i = 0; i < features.getSize(); i++) {
                copy.set(i, features.get(i));
            }
            return copy;
        }
    }
    
    public VisionFeatureCache(int maxCacheSize) {
        this.maxCacheSize = maxCacheSize;
        // Use LinkedHashMap for LRU behavior with synchronized wrapper
        this.cache = new LinkedHashMap<String, CachedFeature>(maxCacheSize + 1, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, CachedFeature> eldest) {
                boolean shouldRemove = size() > maxCacheSize;
                return shouldRemove;
            }
        };
        
    }
    
    /**
     * Default constructor with reasonable cache size.
     */
    public VisionFeatureCache() {
        this(100); // Default to 100 cached images
    }
    
    /**
     * Get features from cache or compute if not cached.
     * 
     * @param imageData The raw image data for hashing
     * @param computeFeatures Supplier to compute features if not cached
     * @return Computed or cached vision features
     */
    public FloatArray getOrCompute(byte[] imageData, Supplier<FloatArray> computeFeatures) {
        String imageHash = computeHash(imageData);
        return getOrCompute(imageHash, computeFeatures);
    }
    
    /**
     * Get features from cache or compute if not cached using provided hash.
     * 
     * @param imageHash Pre-computed hash of the image
     * @param computeFeatures Supplier to compute features if not cached
     * @return Computed or cached vision features
     */
    public synchronized FloatArray getOrCompute(String imageHash, Supplier<FloatArray> computeFeatures) {
        // Check cache first
        CachedFeature cached = cache.get(imageHash);
        if (cached != null) {
            hits.incrementAndGet();
            return cached.getFeaturesCopy();
        }
        
        // Cache miss - compute features
        misses.incrementAndGet();
        
        FloatArray features = computeFeatures.get();
        
        // Add to cache
        cache.put(imageHash, new CachedFeature(features));
        
        return features;
    }
    
    /**
     * Compute SHA-256 hash of image data for cache key.
     */
    private String computeHash(byte[] imageData) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = digest.digest(imageData);
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
            // Fallback to simple hash if SHA-256 not available
            return String.valueOf(java.util.Arrays.hashCode(imageData));
        }
    }
    
    /**
     * Get cache statistics.
     */
    public CacheStats getStats() {
        return new CacheStats(hits.get(), misses.get(), cache.size(), maxCacheSize);
    }
    
    /**
     * Clear all cached features.
     */
    public synchronized void clear() {
        int oldSize = cache.size();
        cache.clear();
        hits.set(0);
        misses.set(0);
    }
    
    /**
     * Remove specific entry from cache.
     */
    public synchronized boolean remove(String imageHash) {
        CachedFeature removed = cache.remove(imageHash);
        if (removed != null) {
            return true;
        }
        return false;
    }
    
    /**
     * Cache statistics holder.
     */
    public static class CacheStats {
        public final long hits;
        public final long misses;
        public final int currentSize;
        public final int maxSize;
        public final double hitRate;
        
        CacheStats(long hits, long misses, int currentSize, int maxSize) {
            this.hits = hits;
            this.misses = misses;
            this.currentSize = currentSize;
            this.maxSize = maxSize;
            this.hitRate = (hits + misses > 0) ? (double) hits / (hits + misses) : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format("CacheStats{hits=%d, misses=%d, hitRate=%.2f%%, size=%d/%d}", 
                               hits, misses, hitRate * 100, currentSize, maxSize);
        }
    }
    
    /**
     * Print cache statistics to stderr.
     */
    public void printStats() {
        CacheStats stats = getStats();
    }
}