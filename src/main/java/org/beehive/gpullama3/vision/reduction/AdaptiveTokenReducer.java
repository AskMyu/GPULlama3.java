package org.beehive.gpullama3.vision.reduction;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Adaptive Token Reducer for LLaVA Vision Processing
 * 
 * Implements token pruning and merging based on LLaVA-PruMerge research (2025).
 * Reduces visual tokens by up to 75% while maintaining quality through importance scoring.
 * 
 * Algorithm:
 * 1. Compute importance scores for all tokens (L2 norm)
 * 2. Identify top-k most important tokens to keep
 * 3. Merge information from pruned tokens into kept tokens
 * 4. Return reduced token set with enhanced feature representations
 */
public class AdaptiveTokenReducer {
    
    // Default configuration based on LLaVA-PruMerge research
    private static final float DEFAULT_PRUNING_RATIO = 0.75f;  // Keep 25% of tokens
    private static final float MERGE_WEIGHT = 0.1f;            // Weight for merging pruned tokens
    private static final boolean ENABLE_SPATIAL_MERGING = true; // Merge spatially adjacent tokens
    
    private final float pruningRatio;
    private final boolean enableMerging;
    private final boolean spatialMerging;
    private final int originalPatchesPerSide; // 24 for 336px CLIP images
    
    // Statistics
    private long totalTokensProcessed = 0;
    private long totalTokensReduced = 0;
    private double averageReductionRatio = 0.0;

    /**
     * Create reducer with default settings (75% reduction)
     */
    public AdaptiveTokenReducer() {
        this(DEFAULT_PRUNING_RATIO, true, ENABLE_SPATIAL_MERGING, 24);
    }
    
    /**
     * Create reducer with custom pruning ratio
     */
    public AdaptiveTokenReducer(float pruningRatio) {
        this(pruningRatio, true, ENABLE_SPATIAL_MERGING, 24);
    }

    /**
     * Create reducer with full configuration
     */
    public AdaptiveTokenReducer(float pruningRatio, boolean enableMerging, 
                               boolean spatialMerging, int originalPatchesPerSide) {
        this.pruningRatio = Math.max(0.0f, Math.min(0.95f, pruningRatio)); // Clamp to [0, 0.95]
        this.enableMerging = enableMerging;
        this.spatialMerging = spatialMerging;
        this.originalPatchesPerSide = originalPatchesPerSide;
        
        System.out.println("AdaptiveTokenReducer initialized:");
        System.out.println("  Pruning ratio: " + (this.pruningRatio * 100) + "%");
        System.out.println("  Merging enabled: " + enableMerging);
        System.out.println("  Spatial merging: " + spatialMerging);
        System.out.println("  Original patches per side: " + originalPatchesPerSide);
    }

    /**
     * Reduce tokens from numTokens to (1-pruningRatio) * numTokens
     */
    public FloatArray reduceTokens(FloatArray tokens, int numTokens, int tokenDim) {
        long startTime = System.nanoTime();
        
        if (pruningRatio <= 0.01f) {
            // No reduction needed
            System.err.printf("[TOKEN-REDUCTION] Skipping reduction (ratio: %.1f%%)%n", pruningRatio * 100);
            return tokens;
        }
        
        int reducedTokenCount = (int)(numTokens * (1.0f - pruningRatio));
        reducedTokenCount = Math.max(1, Math.min(numTokens, reducedTokenCount)); // Ensure valid range
        
        System.err.printf("[TOKEN-REDUCTION] Reducing %d tokens -> %d tokens (%.1f%% reduction)%n", 
                         numTokens, reducedTokenCount, pruningRatio * 100);
        
        // Step 1: Compute importance scores for each token
        float[] importanceScores = computeImportanceScores(tokens, numTokens, tokenDim);
        
        // Step 2: Find indices of top-k most important tokens
        int[] keepIndices = selectTopKTokens(importanceScores, reducedTokenCount);
        
        // Step 3: Create reduced token array
        FloatArray reducedTokens = new FloatArray(reducedTokenCount * tokenDim);
        
        if (enableMerging) {
            // Step 4: Merge pruned token information into kept tokens
            mergeTokensWithPruned(tokens, reducedTokens, keepIndices, 
                                 numTokens, tokenDim, reducedTokenCount, importanceScores);
        } else {
            // Simple copy without merging
            copySelectedTokens(tokens, reducedTokens, keepIndices, tokenDim);
        }
        
        // Update statistics
        updateStatistics(numTokens, reducedTokenCount);
        
        long duration = System.nanoTime() - startTime;
        System.err.printf("[TOKEN-REDUCTION] Reduction completed in %.2f ms%n", duration / 1_000_000.0);
        
        return reducedTokens;
    }

    /**
     * Compute importance score for each token using L2 norm
     */
    private float[] computeImportanceScores(FloatArray tokens, int numTokens, int tokenDim) {
        float[] scores = new float[numTokens];
        
        for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
            float norm = 0.0f;
            int baseIdx = tokenIdx * tokenDim;
            
            // Compute L2 norm of token features
            for (int dim = 0; dim < tokenDim; dim++) {
                float val = tokens.get(baseIdx + dim);
                norm += val * val;
            }
            
            scores[tokenIdx] = (float)Math.sqrt(norm);
            
            // Add spatial importance bonus for center tokens (heuristic)
            if (spatialMerging && originalPatchesPerSide > 0) {
                int row = tokenIdx / originalPatchesPerSide;
                int col = tokenIdx % originalPatchesPerSide;
                int center = originalPatchesPerSide / 2;
                
                // Distance from center (Manhattan distance)
                int distFromCenter = Math.abs(row - center) + Math.abs(col - center);
                float spatialBonus = 1.0f / (1.0f + distFromCenter * 0.1f); // Slight center bias
                
                scores[tokenIdx] *= spatialBonus;
            }
        }
        
        return scores;
    }

    /**
     * Select top-k tokens based on importance scores
     */
    private int[] selectTopKTokens(float[] importanceScores, int k) {
        // Create index-score pairs
        Integer[] indices = new Integer[importanceScores.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // Sort by importance score (descending)
        Arrays.sort(indices, (a, b) -> Float.compare(importanceScores[b], importanceScores[a]));
        
        // Take top k indices and sort them (to maintain spatial order)
        int[] topK = new int[k];
        for (int i = 0; i < k; i++) {
            topK[i] = indices[i];
        }
        Arrays.sort(topK); // Maintain spatial order
        
        return topK;
    }

    /**
     * Copy selected tokens to reduced array (without merging)
     */
    private void copySelectedTokens(FloatArray source, FloatArray dest, int[] keepIndices, int tokenDim) {
        for (int i = 0; i < keepIndices.length; i++) {
            int srcTokenIdx = keepIndices[i];
            int srcBase = srcTokenIdx * tokenDim;
            int destBase = i * tokenDim;
            
            for (int dim = 0; dim < tokenDim; dim++) {
                dest.set(destBase + dim, source.get(srcBase + dim));
            }
        }
    }

    /**
     * Merge pruned tokens into kept tokens for enhanced representation
     */
    private void mergeTokensWithPruned(FloatArray source, FloatArray dest, int[] keepIndices,
                                      int numTokens, int tokenDim, int reducedTokenCount,
                                      float[] importanceScores) {
        // First copy the kept tokens
        copySelectedTokens(source, dest, keepIndices, tokenDim);
        
        // Create set of kept indices for quick lookup
        Set<Integer> keptSet = new HashSet<>();
        for (int idx : keepIndices) {
            keptSet.add(idx);
        }
        
        // For each pruned token, find the nearest kept token and merge
        for (int prunedIdx = 0; prunedIdx < numTokens; prunedIdx++) {
            if (keptSet.contains(prunedIdx)) continue; // Skip kept tokens
            
            // Find nearest kept token (spatially)
            int nearestKeptIdx = findNearestKeptToken(prunedIdx, keepIndices);
            
            if (nearestKeptIdx >= 0 && nearestKeptIdx < reducedTokenCount) {
                // Merge pruned token into nearest kept token
                float mergeWeight = MERGE_WEIGHT * (importanceScores[prunedIdx] / 
                                   (importanceScores[prunedIdx] + importanceScores[keepIndices[nearestKeptIdx]]));
                
                int prunedBase = prunedIdx * tokenDim;
                int mergeBase = nearestKeptIdx * tokenDim;
                
                for (int dim = 0; dim < tokenDim; dim++) {
                    float originalVal = dest.get(mergeBase + dim);
                    float prunedVal = source.get(prunedBase + dim);
                    float merged = originalVal + mergeWeight * prunedVal;
                    dest.set(mergeBase + dim, merged);
                }
            }
        }
    }

    /**
     * Find the nearest kept token to a pruned token (spatially)
     */
    private int findNearestKeptToken(int prunedIdx, int[] keepIndices) {
        if (!spatialMerging || originalPatchesPerSide <= 0) {
            // Return first kept token if no spatial info
            return 0;
        }
        
        int prunedRow = prunedIdx / originalPatchesPerSide;
        int prunedCol = prunedIdx % originalPatchesPerSide;
        
        int nearestIdx = 0;
        int minDistance = Integer.MAX_VALUE;
        
        for (int i = 0; i < keepIndices.length; i++) {
            int keptIdx = keepIndices[i];
            int keptRow = keptIdx / originalPatchesPerSide;
            int keptCol = keptIdx % originalPatchesPerSide;
            
            // Manhattan distance
            int distance = Math.abs(prunedRow - keptRow) + Math.abs(prunedCol - keptCol);
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestIdx = i; // Index in the keepIndices array, not original token index
            }
        }
        
        return nearestIdx;
    }

    /**
     * Update reduction statistics
     */
    private void updateStatistics(int originalTokens, int reducedTokens) {
        totalTokensProcessed += originalTokens;
        totalTokensReduced += (originalTokens - reducedTokens);
        
        double currentReduction = (double)(originalTokens - reducedTokens) / originalTokens;
        averageReductionRatio = (averageReductionRatio + currentReduction) / 2.0;
    }

    /**
     * Get reduction statistics
     */
    public void printStatistics() {
        System.out.println("AdaptiveTokenReducer Statistics:");
        System.out.printf("  Total tokens processed: %d%n", totalTokensProcessed);
        System.out.printf("  Total tokens reduced: %d%n", totalTokensReduced);
        System.out.printf("  Average reduction ratio: %.1f%%%n", averageReductionRatio * 100);
        System.out.printf("  Memory savings: ~%.1fMB per image%n", 
                         (totalTokensReduced * 1024 * 4) / (1024.0 * 1024.0)); // 4 bytes per float
    }

    /**
     * Reset statistics
     */
    public void resetStatistics() {
        totalTokensProcessed = 0;
        totalTokensReduced = 0;
        averageReductionRatio = 0.0;
    }
    
    /**
     * Get current pruning ratio
     */
    public float getPruningRatio() {
        return pruningRatio;
    }
    
    /**
     * Check if merging is enabled
     */
    public boolean isMergingEnabled() {
        return enableMerging;
    }
}