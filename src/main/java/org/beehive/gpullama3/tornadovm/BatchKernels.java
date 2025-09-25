package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * TornadoVM GPU kernels for batch processing operations.
 * These kernels are designed to process multiple tokens simultaneously
 * to maintain context coherence, particularly important for MoE models
 * where batch processing can solve expert routing context isolation.
 */
public class BatchKernels {

    /**
     * Batch embedding lookup kernel.
     * Processes multiple token IDs simultaneously to retrieve their embeddings.
     */
    public static void batchEmbeddingLookup(
            IntArray tokens,
            FloatArray embeddings,
            FloatArray output,
            int batchSize,
            int embeddingDim,
            int vocabSize) {

        for (@Parallel int tokenIdx = 0; tokenIdx < batchSize; tokenIdx++) {
            int tokenId = tokens.get(tokenIdx);

            // Validate token ID
            if (tokenId >= 0 && tokenId < vocabSize) {
                int outputOffset = tokenIdx * embeddingDim;
                int embeddingOffset = tokenId * embeddingDim;

                // Copy embedding vector for this token
                for (int dim = 0; dim < embeddingDim; dim++) {
                    float embeddingValue = embeddings.get(embeddingOffset + dim);
                    output.set(outputOffset + dim, embeddingValue);
                }
            } else {
                // Handle invalid token ID by zeroing the output
                int outputOffset = tokenIdx * embeddingDim;
                for (int dim = 0; dim < embeddingDim; dim++) {
                    output.set(outputOffset + dim, 0.0f);
                }
            }
        }
    }

    /**
     * Batch RMS normalization kernel.
     */
    public static void batchRMSNorm(
            FloatArray input,
            FloatArray weights,
            FloatArray output,
            int batchSize,
            int embeddingDim,
            float epsilon) {

        for (@Parallel int tokenIdx = 0; tokenIdx < batchSize; tokenIdx++) {
            int tokenOffset = tokenIdx * embeddingDim;

            // Compute sum of squares for this token
            float sumSquares = 0.0f;
            for (int dim = 0; dim < embeddingDim; dim++) {
                float value = input.get(tokenOffset + dim);
                sumSquares += value * value;
            }

            // Compute RMS normalization factor
            float rms = (float) Math.sqrt(sumSquares / embeddingDim + epsilon);
            float invRms = 1.0f / rms;

            // Apply normalization with weights
            for (int dim = 0; dim < embeddingDim; dim++) {
                float inputValue = input.get(tokenOffset + dim);
                float weight = weights.get(dim);
                float normalizedValue = inputValue * invRms * weight;
                output.set(tokenOffset + dim, normalizedValue);
            }
        }
    }

    /**
     * Batch matrix multiplication kernel.
     */
    public static void batchMatrixMultiply(
            FloatArray input,
            FloatArray weights,
            FloatArray output,
            int batchSize,
            int inputDim,
            int outputDim) {

        for (@Parallel int globalId = 0; globalId < batchSize * outputDim; globalId++) {
            int tokenIdx = globalId / outputDim;
            int outputIdx = globalId % outputDim;

            if (tokenIdx < batchSize && outputIdx < outputDim) {
                int inputOffset = tokenIdx * inputDim;
                int outputOffset = tokenIdx * outputDim;
                int weightOffset = outputIdx * inputDim;

                float sum = 0.0f;
                for (int i = 0; i < inputDim; i++) {
                    float inputValue = input.get(inputOffset + i);
                    float weightValue = weights.get(weightOffset + i);
                    sum += inputValue * weightValue;
                }

                output.set(outputOffset + outputIdx, sum);
            }
        }
    }

    /**
     * Batch SwiGLU activation kernel for MoE models.
     */
    public static void batchSwiGLU(
            FloatArray gateProjections,
            FloatArray upProjections,
            FloatArray output,
            int batchSize,
            int hiddenDim) {

        for (@Parallel int globalId = 0; globalId < batchSize * hiddenDim; globalId++) {
            int tokenIdx = globalId / hiddenDim;
            int dim = globalId % hiddenDim;

            if (tokenIdx < batchSize && dim < hiddenDim) {
                int offset = tokenIdx * hiddenDim + dim;

                float gate = gateProjections.get(offset);
                float up = upProjections.get(offset);

                // Apply SiLU (Swish) activation to gate
                float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
                float silu = gate * sigmoid;

                // SwiGLU: silu(gate) * up
                float result = silu * up;

                output.set(offset, result);
            }
        }
    }

    /**
     * Batch expert routing kernel for MoE models.
     */
    public static void batchExpertRouting(
            FloatArray input,
            FloatArray routerWeights,
            FloatArray routerLogits,
            int batchSize,
            int hiddenDim,
            int numExperts) {

        for (@Parallel int globalId = 0; globalId < batchSize * numExperts; globalId++) {
            int tokenIdx = globalId / numExperts;
            int expertIdx = globalId % numExperts;

            if (tokenIdx < batchSize && expertIdx < numExperts) {
                int inputOffset = tokenIdx * hiddenDim;
                int outputOffset = tokenIdx * numExperts + expertIdx;

                // Compute router logit: input · router_weights[:, expert_idx]
                float logit = 0.0f;
                for (int dim = 0; dim < hiddenDim; dim++) {
                    float inputValue = input.get(inputOffset + dim);
                    float weightValue = routerWeights.get(dim * numExperts + expertIdx);
                    logit += inputValue * weightValue;
                }

                routerLogits.set(outputOffset, logit);
            }
        }
    }

    /**
     * Utility method to create optimal worker grid for batch operations.
     */
    public static WorkerGrid2D createBatchWorkerGrid(int batchSize, int dimension) {
        // Optimize grid dimensions for GPU architecture
        int gridX = Math.min(batchSize, 256); // Max 256 threads per block in X dimension
        int gridY = Math.min(dimension, 1024 / gridX); // Stay within total thread limit
        return new WorkerGrid2D(gridX, gridY);
    }

    /**
     * Utility method to create worker grid for 1D batch operations.
     */
    public static WorkerGrid1D createBatchWorkerGrid1D(int totalElements) {
        // Optimize for GPU warp size (32) and block size limits
        int gridSize = Math.min(totalElements, 1024);
        return new WorkerGrid1D(gridSize);
    }
}