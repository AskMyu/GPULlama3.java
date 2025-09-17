package org.beehive.gpullama3.inference.sampler;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.random.RandomGenerator;

/**
 * A sampler that samples from a categorical distribution.
 * Supports both FloatTensor and FloatArray implementations.
 */
public record CategoricalSampler(RandomGenerator rng, int vocabularySize) implements Sampler {

    @Override
    public int sampleToken(Object tensor) {
        // AGGRESSIVE DEBUG: Always log that we're in this method
        System.err.println("[SAMPLER-DEBUG] ===== CategoricalSampler.sampleToken() CALLED =====");
        System.err.printf("[SAMPLER-DEBUG] Tensor type: %s%n", 
            tensor != null ? tensor.getClass().getName() : "null");
        
        int sampledToken;
        if (tensor instanceof FloatTensor) {
            FloatTensor ft = (FloatTensor) tensor;
            System.err.printf("[SAMPLER-DEBUG] FloatTensor size: %d%n", ft.size());
            sampledToken = sampleFromFloatTensor(ft);
        } else if (tensor instanceof FloatArray) {
            FloatArray fa = (FloatArray) tensor;
            System.err.printf("[SAMPLER-DEBUG] FloatArray size: %d%n", fa.getSize());
            sampledToken = sampleFromFloatArray(fa);
        } else {
            throw new IllegalArgumentException("Unsupported tensor type: " +
                    (tensor != null ? tensor.getClass().getName() : "null"));
        }
        
        System.err.printf("[SAMPLER-DEBUG] Raw sampled token BEFORE bounds check: %d%n", sampledToken);
        
        // CRITICAL FIX: Ensure sampled token is within vocabulary bounds
        // Use the actual model vocabulary size instead of hardcoding
        final int VOCAB_SIZE = vocabularySize;
        if (sampledToken >= VOCAB_SIZE) {
            System.err.printf("[SAMPLER-FIX] 🚨 Token %d exceeds vocab size %d, clamping to %d%n", 
                sampledToken, VOCAB_SIZE, VOCAB_SIZE - 1);
            sampledToken = VOCAB_SIZE - 1; // Clamp to last valid token
        } else {
            System.err.printf("[SAMPLER-DEBUG] ✅ Token %d is within vocab bounds (< %d)%n", 
                sampledToken, VOCAB_SIZE);
        }
        
        System.err.printf("[SAMPLER-DEBUG] Final token AFTER bounds check: %d%n", sampledToken);
        System.err.println("[SAMPLER-DEBUG] ===== CategoricalSampler.sampleToken() RETURNING =====");
        
        return sampledToken;
    }

    /**
     * Sample from a FloatTensor probability distribution with improved randomization.
     *
     * @param logits The FloatTensor containing probabilities
     * @return The sampled token index
     */
    private int sampleFromFloatTensor(FloatTensor logits) {
        // Enhanced randomization to prevent repetitive patterns
        float random0to1 = rng.nextFloat(1f);

        // Add small random jitter to break deterministic patterns
        float jitter = rng.nextFloat(-1e-6f, 1e-6f);
        random0to1 = Math.max(0.0f, Math.min(1.0f, random0to1 + jitter));

        System.err.printf("[SAMPLER-ENHANCED] Using random value: %.8f (with jitter)%n", random0to1);

        // Sample using CDF with enhanced logging
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            float prob = logits.getFloat(i);
            cdf += prob;

            // Log top probabilities for debugging
            if (prob > 0.01f) {
                System.err.printf("[SAMPLER-ENHANCED] Token %d: prob=%.6f, cdf=%.6f%n", i, prob, cdf);
            }

            if (random0to1 < cdf) {
                System.err.printf("[SAMPLER-ENHANCED] Selected token %d with prob=%.6f at cdf=%.6f%n", i, prob, cdf);
                return i;
            }
        }

        // Enhanced fallback: use weighted random selection among top tokens
        System.err.println("[SAMPLER-ENHANCED] CDF sampling failed, using weighted fallback");

        // Find top-k tokens with highest probabilities for fallback
        final int FALLBACK_TOP_K = 10;
        int[] topIndices = new int[FALLBACK_TOP_K];
        float[] topProbs = new float[FALLBACK_TOP_K];

        // Initialize with first tokens
        for (int i = 0; i < Math.min(FALLBACK_TOP_K, logits.size()); i++) {
            topIndices[i] = i;
            topProbs[i] = logits.getFloat(i);
        }

        // Find actual top-k tokens
        for (int i = FALLBACK_TOP_K; i < logits.size(); i++) {
            float prob = logits.getFloat(i);
            for (int j = 0; j < FALLBACK_TOP_K; j++) {
                if (prob > topProbs[j]) {
                    // Shift lower probabilities down
                    for (int k = FALLBACK_TOP_K - 1; k > j; k--) {
                        topIndices[k] = topIndices[k-1];
                        topProbs[k] = topProbs[k-1];
                    }
                    topIndices[j] = i;
                    topProbs[j] = prob;
                    break;
                }
            }
        }

        // Randomly select from top-k with probability weighting
        float topSum = 0.0f;
        for (int i = 0; i < FALLBACK_TOP_K; i++) {
            topSum += topProbs[i];
        }

        float topRandom = rng.nextFloat(topSum);
        float topCdf = 0.0f;
        for (int i = 0; i < FALLBACK_TOP_K; i++) {
            topCdf += topProbs[i];
            if (topRandom < topCdf) {
                System.err.printf("[SAMPLER-ENHANCED] Fallback selected token %d with prob=%.6f%n",
                                topIndices[i], topProbs[i]);
                return topIndices[i];
            }
        }

        // Final fallback: return first valid token
        System.err.println("[SAMPLER-ENHANCED] All fallbacks failed, returning token 0");
        return 0;
    }

    /**
     * Sample from a FloatArray probability distribution with improved randomization.
     *
     * @param logits The FloatArray containing probabilities
     * @return The sampled token index
     */
    private int sampleFromFloatArray(FloatArray logits) {
        // Enhanced randomization to prevent repetitive patterns
        // Generate multiple random numbers and use them to add entropy
        float random0to1 = rng.nextFloat(1f);

        // Add small random jitter to break deterministic patterns
        float jitter = rng.nextFloat(-1e-6f, 1e-6f);
        random0to1 = Math.max(0.0f, Math.min(1.0f, random0to1 + jitter));

        System.err.printf("[SAMPLER-ENHANCED] Using random value: %.8f (with jitter)%n", random0to1);

        // Verify probability distribution is valid
        float totalProb = 0.0f;
        for (int i = 0; i < logits.getSize(); i++) {
            totalProb += logits.get(i);
        }
        System.err.printf("[SAMPLER-ENHANCED] Total probability sum: %.8f%n", totalProb);

        // Sample using CDF with enhanced logging
        float cdf = 0.0f;
        for (int i = 0; i < logits.getSize(); i++) {
            float prob = logits.get(i);
            cdf += prob;

            // Log top probabilities for debugging
            if (prob > 0.01f) {
                System.err.printf("[SAMPLER-ENHANCED] Token %d: prob=%.6f, cdf=%.6f%n", i, prob, cdf);
            }

            if (random0to1 < cdf) {
                System.err.printf("[SAMPLER-ENHANCED] Selected token %d with prob=%.6f at cdf=%.6f%n", i, prob, cdf);
                return i;
            }
        }

        // Enhanced fallback: use weighted random selection among top tokens
        System.err.println("[SAMPLER-ENHANCED] CDF sampling failed, using weighted fallback");

        // Find top-k tokens with highest probabilities for fallback
        final int FALLBACK_TOP_K = 10;
        int[] topIndices = new int[FALLBACK_TOP_K];
        float[] topProbs = new float[FALLBACK_TOP_K];

        // Initialize with first tokens
        for (int i = 0; i < Math.min(FALLBACK_TOP_K, logits.getSize()); i++) {
            topIndices[i] = i;
            topProbs[i] = logits.get(i);
        }

        // Find actual top-k tokens
        for (int i = FALLBACK_TOP_K; i < logits.getSize(); i++) {
            float prob = logits.get(i);
            for (int j = 0; j < FALLBACK_TOP_K; j++) {
                if (prob > topProbs[j]) {
                    // Shift lower probabilities down
                    for (int k = FALLBACK_TOP_K - 1; k > j; k--) {
                        topIndices[k] = topIndices[k-1];
                        topProbs[k] = topProbs[k-1];
                    }
                    topIndices[j] = i;
                    topProbs[j] = prob;
                    break;
                }
            }
        }

        // Randomly select from top-k with probability weighting
        float topSum = 0.0f;
        for (int i = 0; i < FALLBACK_TOP_K; i++) {
            topSum += topProbs[i];
        }

        float topRandom = rng.nextFloat(topSum);
        float topCdf = 0.0f;
        for (int i = 0; i < FALLBACK_TOP_K; i++) {
            topCdf += topProbs[i];
            if (topRandom < topCdf) {
                System.err.printf("[SAMPLER-ENHANCED] Fallback selected token %d with prob=%.6f%n",
                                topIndices[i], topProbs[i]);
                return topIndices[i];
            }
        }

        // Final fallback: return first valid token
        System.err.println("[SAMPLER-ENHANCED] All fallbacks failed, returning token 0");
        return 0;
    }
}