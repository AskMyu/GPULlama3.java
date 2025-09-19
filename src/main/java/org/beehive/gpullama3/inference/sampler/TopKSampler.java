package org.beehive.gpullama3.inference.sampler;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Top-K sampling implementation supporting both FloatTensor and FloatArray.
 * Samples from the K most likely tokens, optimized for Gemma models.
 */
public final class TopKSampler implements Sampler {

    private final int[] indices;
    private final float[] values;
    private final int topK;
    private final RandomGenerator rng;
    private final int vocabularySize;

    public TopKSampler(int vocabularySize, int topK, RandomGenerator rng) {
        this.vocabularySize = vocabularySize;
        this.topK = Math.min(topK, vocabularySize);
        this.rng = rng;
        this.indices = new int[vocabularySize];
        this.values = new float[vocabularySize];

        // Initialize indices array
        for (int i = 0; i < vocabularySize; i++) {
            indices[i] = i;
        }
    }

    @Override
    public int sampleToken(Object logits) {
        if (logits instanceof FloatTensor tensor) {
            return sampleFromTensor(tensor);
        } else if (logits instanceof FloatArray array) {
            return sampleFromArray(array);
        } else {
            throw new IllegalArgumentException("Unsupported logits type: " + logits.getClass());
        }
    }

    private int sampleFromTensor(FloatTensor logits) {
        // Copy logits to values array
        for (int i = 0; i < vocabularySize; i++) {
            values[i] = logits.getFloat(i);
            indices[i] = i;
        }

        return performTopKSampling();
    }

    private int sampleFromArray(FloatArray logits) {
        // Copy logits to values array
        for (int i = 0; i < vocabularySize; i++) {
            values[i] = logits.get(i);
            indices[i] = i;
        }

        return performTopKSampling();
    }

    private int performTopKSampling() {
        // Partial sort to find top-K elements using selection algorithm
        // This is more efficient than full sort for small K
        for (int i = 0; i < topK; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < vocabularySize; j++) {
                if (values[indices[j]] > values[indices[maxIdx]]) {
                    maxIdx = j;
                }
            }
            // Swap to bring max element to position i
            if (maxIdx != i) {
                int temp = indices[i];
                indices[i] = indices[maxIdx];
                indices[maxIdx] = temp;
            }
        }

        // Convert top-K logits to probabilities using softmax
        float maxLogit = values[indices[0]];
        double sum = 0.0;
        float[] probs = new float[topK];

        for (int i = 0; i < topK; i++) {
            float logit = values[indices[i]];
            probs[i] = (float) Math.exp(logit - maxLogit);
            sum += probs[i];
        }

        // Normalize probabilities
        for (int i = 0; i < topK; i++) {
            probs[i] /= sum;
        }

        // Sample from top-K distribution
        float randomValue = rng.nextFloat();
        float cumulative = 0.0f;

        // DEBUG: Log top-K sampling details
        System.err.printf("[TOPK-DEBUG] Sampling from top-%d tokens, random=%.6f%n", topK, randomValue);
        System.err.printf("[TOPK-DEBUG] Top 5 tokens: [%d(%.4f), %d(%.4f), %d(%.4f), %d(%.4f), %d(%.4f)]%n",
            indices[0], probs[0], indices[1], probs[1], indices[2], probs[2], indices[3], probs[3], indices[4], probs[4]);

        for (int i = 0; i < topK; i++) {
            cumulative += probs[i];
            if (randomValue <= cumulative) {
                System.err.printf("[TOPK-DEBUG] Selected token: %d (index %d, prob=%.4f, cumulative=%.4f)%n", indices[i], i, probs[i], cumulative);
                return indices[i];
            }
        }

        // Fallback to last token if rounding errors occur
        return indices[topK - 1];
    }
}