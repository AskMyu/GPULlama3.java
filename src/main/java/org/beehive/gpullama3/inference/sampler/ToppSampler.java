package org.beehive.gpullama3.inference.sampler;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.Comparator;
import java.util.random.RandomGenerator;

/**
 * Top-p sampling (nucleus sampling) implementation supporting both FloatTensor and FloatArray.
 * Samples from the smallest set of tokens that exceed probability topp.
 */
public final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;
    final int vocabularySize;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
        this.vocabularySize = maxNumberOfElements;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    @Override
    public int sampleToken(Object tensor) {
        if (tensor instanceof FloatTensor) {
            return sampleFromFloatTensor((FloatTensor) tensor);
        } else if (tensor instanceof FloatArray) {
            return sampleFromFloatArray((FloatArray) tensor);
        }
        throw new IllegalArgumentException("Unsupported tensor type: " +
                (tensor != null ? tensor.getClass().getName() : "null"));
    }

    /**
     * Implementation of top-p sampling for FloatTensor.
     */
    private int sampleFromFloatTensor(FloatTensor logits) {
        // Create a comparator that compares indices based on their values in the tensor
        Comparator<Integer> comparator = Comparator.comparingDouble(logits::getFloat).reversed();

        int n = logits.size();
        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        // CRITICAL FIX: Use actual tensor size instead of indices.length to prevent bounds exception
        for (int i = 0; i < n; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        return processTopP(logits, comparator, head);
    }

    /**
     * Implementation of top-p sampling for FloatArray.
     */
    private int sampleFromFloatArray(FloatArray logits) {
        // Create a comparator that compares indices based on their values in the array
        Comparator<Integer> comparator = (a, b) -> Float.compare(logits.get(b), logits.get(a)); // reversed order

        int n = logits.getSize();
        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        // CRITICAL FIX: Use actual array size instead of indices.length to prevent bounds exception
        for (int i = 0; i < n; i++) {
            if (logits.get(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        return processTopP(logits, comparator, head);
    }

    /**
     * Common implementation for processing top-p sampling once indices are prepared.
     * Uses a type-specific value getter function to access tensor values.
     */
    private int processTopP(Object logits, Comparator<Integer> comparator, int n0) {
        // CRITICAL FIX: Ensure all vocab bounds checking
        final int VOCAB_SIZE = vocabularySize;
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements exceeds topp
        // O(k lg n0)
        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);

            float value;
            if (logits instanceof FloatTensor) {
                value = ((FloatTensor) logits).getFloat(indices[i]);
            } else {
                value = ((FloatArray) logits).get(indices[i]);
            }

            cumulativeProb += value;
            if (cumulativeProb > topp) {
                lastIndex = i;
                break; // we've exceeded topp by including lastIndex
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            float value;
            if (logits instanceof FloatTensor) {
                value = ((FloatTensor) logits).getFloat(indices[i]);
            } else {
                value = ((FloatArray) logits).get(indices[i]);
            }

            cdf += value;
            if (r < cdf) {
                int token = indices[i];
                // CRITICAL FIX: Ensure token is within vocabulary bounds
                if (token < VOCAB_SIZE) {
                    return token;
                }
                // If invalid token, continue to find a valid one
            }
        }

        // CRITICAL FIX: Don't return potentially invalid tokens
        // Find valid token with highest probability in case of rounding errors
        int bestToken = -1;
        float bestProb = -1.0f;
        
        for (int i = n0 - 1; i >= lastIndex; i--) {
            int token = indices[i];
            if (token < VOCAB_SIZE) { // Only consider valid tokens
                float value;
                if (logits instanceof FloatTensor) {
                    value = ((FloatTensor) logits).getFloat(token);
                } else {
                    value = ((FloatArray) logits).get(token);
                }
                
                if (bestToken == -1 || value > bestProb) {
                    bestToken = token;
                    bestProb = value;
                }
            }
        }
        
        // If no valid token found, use safe fallback (space token or 0)
        return bestToken != -1 ? bestToken : 0;
    }
}