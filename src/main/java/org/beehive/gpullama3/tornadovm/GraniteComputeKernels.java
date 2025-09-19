package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Granite-specific GPU compute kernels for TornadoVM.
 *
 * Implements native GPU kernels for:
 * - Group-Query Attention (GQA) with 32 Q heads and 8 KV heads
 * - SwiGLU activation function
 * - Granite-optimized RoPE rotation
 * - GQA-specific cache operations
 */
public class GraniteComputeKernels {

    /**
     * RoPE rotation optimized for Granite GQA architecture.
     * Handles different head counts for Q (32) and KV (8) tensors.
     */
    public static void ropeRotationGQA(KernelContext context, IntArray positionHolder,
                                       FloatArray q, FloatArray k,
                                       int nHeads, int nKVHeads, int headSize) {
        // Get current position
        int position = positionHolder.get(0);

        // Thread index for parallel execution
        int idx = context.globalIdx;

        // Total elements to process: Q heads + KV heads
        int totalElements = (nHeads + nKVHeads) * headSize / 2;

        if (idx >= totalElements) return;

        // Determine if we're processing Q or K tensor
        boolean isQTensor = idx < (nHeads * headSize / 2);

        int headIdx, elementIdx, tensorIdx;

        if (isQTensor) {
            // Processing Q tensor
            headIdx = idx / (headSize / 2);
            elementIdx = idx % (headSize / 2);
            tensorIdx = headIdx * headSize + elementIdx;
        } else {
            // Processing K tensor
            int kIdx = idx - (nHeads * headSize / 2);
            headIdx = kIdx / (headSize / 2);
            elementIdx = kIdx % (headSize / 2);
            tensorIdx = headIdx * headSize + elementIdx;
        }

        // RoPE computation
        float ropeTheta = 10000.0f;
        float freq = 1.0f / (float) Math.pow(ropeTheta, (2.0f * elementIdx) / headSize);
        float angle = position * freq;
        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);

        // Apply rotation
        int idx1 = tensorIdx;
        int idx2 = tensorIdx + headSize / 2;

        if (isQTensor) {
            float v1 = q.get(idx1);
            float v2 = q.get(idx2);
            q.set(idx1, v1 * cos - v2 * sin);
            q.set(idx2, v1 * sin + v2 * cos);
        } else {
            float v1 = k.get(idx1);
            float v2 = k.get(idx2);
            k.set(idx1, v1 * cos - v2 * sin);
            k.set(idx2, v1 * sin + v2 * cos);
        }
    }

    /**
     * Cache K and V tensors with layout compatible with FlashAttention kernel.
     * CRITICAL: Must match the indexing used in processHeadsFlashAttention:
     * keyOffset = loff + t * kvDim + kvHeadIdx * headSize
     */
    public static void copyToCacheGQA(KernelContext context, FloatArray keyCache, FloatArray k,
                                      FloatArray valueCache, FloatArray v,
                                      IntArray positionHolder, int nKVHeads, int headSize,
                                      int layerIndex, int contextLength) {
        int position = positionHolder.get(0);
        int idx = context.globalIdx;

        // Total KV elements to cache
        int kvDim = nKVHeads * headSize; // This equals config.kvDim()
        int totalKVElements = kvDim;

        if (idx >= totalKVElements) return;

        // Map flat index to kvHead and element
        int kvHead = idx / headSize;
        int element = idx % headSize;

        // Source index in current K/V tensors (same layout as input)
        int srcIdx = kvHead * headSize + element;

        // CRITICAL FIX: Destination index must match FlashAttention expectations
        // FlashAttention uses: loff + t * kvDim + kvHeadIdx * headSize + element
        int loff = layerIndex * contextLength * kvDim;
        int dstIdx = loff + position * kvDim + kvHead * headSize + element;

        // Copy to caches with correct layout
        keyCache.set(dstIdx, k.get(srcIdx));
        valueCache.set(dstIdx, v.get(srcIdx));
    }

    /**
     * Group-Query Attention implementation for GPU.
     * 32 Q heads share 8 KV heads (4:1 ratio).
     */
    public static void processHeadsGQA(KernelContext context,
                                       FloatArray q, FloatArray keyCache, FloatArray valueCache,
                                       FloatArray output,
                                       int nHeads, int nKVHeads, int headSize,
                                       IntArray positionHolder, int layerIndex, int contextLength) {
        int headIdx = context.globalIdx;

        if (headIdx >= nHeads) return;

        int position = positionHolder.get(0);
        int kvMul = nHeads / nKVHeads; // 4 for Granite (32/8)
        int kvHead = headIdx / kvMul; // Which KV head this Q head uses

        // Layer and head offsets
        int layerOffset = layerIndex * contextLength * nKVHeads * headSize;
        int qHeadOffset = headIdx * headSize;

        // Temporary storage for attention scores (using output buffer)
        int attentionOffset = headIdx * contextLength;

        // Compute attention scores
        for (int t = 0; t <= position; t++) {
            float score = 0.0f;

            // Dot product Q[headIdx] · K[kvHead][t]
            for (int i = 0; i < headSize; i++) {
                float qVal = q.get(qHeadOffset + i);

                // GQA cache layout: [layer][position][kvHead][element]
                int kIdx = layerOffset + t * nKVHeads * headSize + kvHead * headSize + i;
                float kVal = keyCache.get(kIdx);

                score += qVal * kVal;
            }

            // Scale by sqrt(headSize) and store
            score /= (float) Math.sqrt(headSize);
            output.set(attentionOffset + t, score);
        }

        // Apply softmax to attention scores
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int t = 0; t <= position; t++) {
            float score = output.get(attentionOffset + t);
            if (score > maxScore) maxScore = score;
        }

        float sumExp = 0.0f;
        for (int t = 0; t <= position; t++) {
            float score = output.get(attentionOffset + t);
            float expScore = (float) Math.exp(score - maxScore);
            output.set(attentionOffset + t, expScore);
            sumExp += expScore;
        }

        // Normalize
        if (sumExp > 0.0f) {
            for (int t = 0; t <= position; t++) {
                float weight = output.get(attentionOffset + t) / sumExp;
                output.set(attentionOffset + t, weight);
            }
        }

        // Weighted sum of values
        for (int i = 0; i < headSize; i++) {
            float result = 0.0f;

            for (int t = 0; t <= position; t++) {
                float weight = output.get(attentionOffset + t);

                // GQA cache layout: [layer][position][kvHead][element]
                int vIdx = layerOffset + t * nKVHeads * headSize + kvHead * headSize + i;
                float vVal = valueCache.get(vIdx);

                result += weight * vVal;
            }

            // Store final output
            output.set(qHeadOffset + i, result);
        }
    }

    /**
     * SwiGLU activation function: silu(gate) * up
     * Standard formulation used in Llama, Mistral, and other modern models
     */
    public static void fusedFeedForwardWithSwiGLU(KernelContext context,
                                                   FloatArray input, FloatArray output,
                                                   FloatArray gateWeights, FloatArray upWeights,
                                                   int inputDim, int hiddenDim, int workGroupSize) {
        int idx = context.globalIdx;

        if (idx >= hiddenDim) return;

        // Compute gate projection (w1 * input)
        float gate = 0.0f;
        for (int i = 0; i < inputDim; i++) {
            gate += gateWeights.get(idx * inputDim + i) * input.get(i);
        }

        // Compute up projection (w3 * input)
        float up = 0.0f;
        for (int i = 0; i < inputDim; i++) {
            up += upWeights.get(idx * inputDim + i) * input.get(i);
        }

        // Apply SiLU to gate projection: gate * sigmoid(gate)
        float siluGate;
        if (gate < -20.0f) {
            siluGate = 0.0f;
        } else if (gate > 20.0f) {
            siluGate = gate;
        } else {
            siluGate = gate / (1.0f + (float) Math.exp(-gate));
        }

        // SwiGLU: silu(gate) * up (standard formulation)
        output.set(idx, siluGate * up);
    }
}