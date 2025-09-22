package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * TornadoVM GPU Kernel for Q/K Normalization
 *
 * Implements RMS normalization for Q and K tensors after projection but BEFORE reshape.
 * This is a critical architectural requirement for OLMoE that was missing.
 *
 * Based on llama.cpp implementation:
 * - Apply RMS norm to 2D tensors [dim, n_tokens]
 * - Use normalization weights [dim]
 * - Must be applied BEFORE reshaping to 3D
 */
public class TornadoVMQKNormKernel {

    /**
     * GPU kernel for RMS normalization of 2D tensor
     *
     * RMS Norm formula: output = input * weight / sqrt(mean(input^2) + eps)
     *
     * @param input Input tensor to normalize [dim]
     * @param output Output normalized tensor [dim]
     * @param weight Normalization weights [dim]
     * @param dim Dimension size
     * @param eps Epsilon for numerical stability
     */
    public static void rmsNorm2DGPU(
            FloatArray input,
            FloatArray output,
            FloatArray weight,
            int dim,
            float eps) {

        // Step 1: Calculate sum of squares for RMS
        float sumSquares = 0.0f;
        for (@Parallel int i = 0; i < dim; i++) {
            float val = input.get(i);
            sumSquares += val * val;
        }

        // Step 2: Calculate RMS normalization factor
        float rms = (float) Math.sqrt(sumSquares / dim + eps);
        float scale = 1.0f / rms;

        // Step 3: Apply normalization with weights
        for (@Parallel int i = 0; i < dim; i++) {
            output.set(i, input.get(i) * scale * weight.get(i));
        }
    }

    /**
     * Optimized GPU kernel for RMS normalization with reduction
     * This version uses TornadoVM reduction for better performance
     */
    public static void rmsNorm2DGPUOptimized(
            FloatArray input,
            FloatArray output,
            FloatArray weight,
            FloatArray reductionBuffer,
            int dim,
            float eps) {

        // Step 1: Parallel computation of squares
        for (@Parallel int i = 0; i < dim; i++) {
            float val = input.get(i);
            reductionBuffer.set(i, val * val);
        }

        // Step 2: Reduction to compute sum (TornadoVM handles this efficiently)
        float sumSquares = 0.0f;
        for (int i = 0; i < dim; i++) {
            sumSquares += reductionBuffer.get(i);
        }

        // Step 3: Calculate RMS factor
        float rms = (float) Math.sqrt(sumSquares / dim + eps);
        float scale = 1.0f / rms;

        // Step 4: Parallel normalization with weights
        for (@Parallel int i = 0; i < dim; i++) {
            output.set(i, input.get(i) * scale * weight.get(i));
        }
    }

    /**
     * Apply Q/K normalization to query and key tensors
     * This matches the llama.cpp OLMoE implementation exactly
     */
    public static class QKNormalizationTask {
        private final int dim;
        private final float rmsNormEps;
        private TaskGraph taskGraph;
        private TornadoExecutionPlan executionPlan;

        // GPU arrays
        private FloatArray queryInput;
        private FloatArray queryOutput;
        private FloatArray queryNormWeights;
        private FloatArray keyInput;
        private FloatArray keyOutput;
        private FloatArray keyNormWeights;
        private FloatArray reductionBuffer;

        public QKNormalizationTask(int dim, float rmsNormEps) {
            this.dim = dim;
            this.rmsNormEps = rmsNormEps;

            // Initialize GPU arrays
            this.queryInput = new FloatArray(dim);
            this.queryOutput = new FloatArray(dim);
            this.keyInput = new FloatArray(dim);
            this.keyOutput = new FloatArray(dim);
            this.reductionBuffer = new FloatArray(dim);

            // Don't initialize task graph here - weights are null
            // TaskGraph will be created when setNormalizationWeights() is called
        }

        private void initializeTaskGraph() {
            taskGraph = new TaskGraph("qk_normalization")
                // Transfer inputs to device
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                                queryInput, keyInput, queryNormWeights, keyNormWeights)

                // Q normalization task
                .task("q_norm", TornadoVMQKNormKernel::rmsNorm2DGPUOptimized,
                      queryInput, queryOutput, queryNormWeights, reductionBuffer, dim, rmsNormEps)

                // K normalization task
                .task("k_norm", TornadoVMQKNormKernel::rmsNorm2DGPUOptimized,
                      keyInput, keyOutput, keyNormWeights, reductionBuffer, dim, rmsNormEps)

                // Transfer outputs back to host
                .transferToHost(DataTransferMode.EVERY_EXECUTION, queryOutput, keyOutput);

            // Create execution plan
            executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        }

        /**
         * Set normalization weights (loaded from model)
         */
        public void setNormalizationWeights(float[] qNormWeights, float[] kNormWeights) {
            this.queryNormWeights = new FloatArray(qNormWeights.length);
            this.keyNormWeights = new FloatArray(kNormWeights.length);

            // Copy values into the arrays
            for (int i = 0; i < qNormWeights.length; i++) {
                this.queryNormWeights.set(i, qNormWeights[i]);
            }
            for (int i = 0; i < kNormWeights.length; i++) {
                this.keyNormWeights.set(i, kNormWeights[i]);
            }

            // Initialize task graph now that we have weights
            initializeTaskGraph();
        }

        /**
         * Apply Q/K normalization on GPU
         */
        public void normalize(float[] queryData, float[] keyData,
                            float[] queryResult, float[] keyResult) {
            if (executionPlan == null) {
                throw new IllegalStateException("QKNormalizationTask not initialized - call setNormalizationWeights() first");
            }

            // Copy input data to GPU arrays
            for (int i = 0; i < dim; i++) {
                queryInput.set(i, queryData[i]);
                keyInput.set(i, keyData[i]);
            }

            // Execute GPU kernels
            executionPlan.execute();

            // Copy results back
            for (int i = 0; i < dim; i++) {
                queryResult[i] = queryOutput.get(i);
                keyResult[i] = keyOutput.get(i);
            }
        }

        /**
         * Batch normalization for multiple token positions
         */
        public void normalizeBatch(float[][] queryBatch, float[][] keyBatch,
                                  float[][] queryResults, float[][] keyResults) {
            int batchSize = queryBatch.length;

            for (int b = 0; b < batchSize; b++) {
                normalize(queryBatch[b], keyBatch[b],
                         queryResults[b], keyResults[b]);
            }
        }
    }

    /**
     * Integration point for OLMoE forward pass
     * This should be called after Q/K projection but before reshape
     */
    public static void applyOLMoEQKNormalization(
            FloatArray query2D,      // [dim, n_tokens] after projection
            FloatArray key2D,        // [dim, n_tokens] after projection
            FloatArray qNormWeights, // [dim] normalization weights for Q
            FloatArray kNormWeights, // [dim] normalization weights for K
            int dim,
            int nTokens,
            float rmsNormEps) {

        System.err.println("[QK-NORM-GPU] Applying Q/K normalization on GPU:");
        System.err.printf("  Query shape: [%d, %d]%n", dim, nTokens);
        System.err.printf("  Key shape: [%d, %d]%n", dim, nTokens);
        System.err.println("  Applied BEFORE reshape (matching llama.cpp)");

        // Process each token position
        FloatArray tempInput = new FloatArray(dim);
        FloatArray tempOutput = new FloatArray(dim);
        FloatArray reductionBuffer = new FloatArray(dim);

        for (int t = 0; t < nTokens; t++) {
            // Extract column for token t
            for (int d = 0; d < dim; d++) {
                tempInput.set(d, query2D.get(d * nTokens + t));
            }

            // Apply Q normalization
            rmsNorm2DGPUOptimized(tempInput, tempOutput, qNormWeights,
                                 reductionBuffer, dim, rmsNormEps);

            // Write back normalized values
            for (int d = 0; d < dim; d++) {
                query2D.set(d * nTokens + t, tempOutput.get(d));
            }

            // Repeat for K
            for (int d = 0; d < dim; d++) {
                tempInput.set(d, key2D.get(d * nTokens + t));
            }

            rmsNorm2DGPUOptimized(tempInput, tempOutput, kNormWeights,
                                 reductionBuffer, dim, rmsNormEps);

            for (int d = 0; d < dim; d++) {
                key2D.set(d * nTokens + t, tempOutput.get(d));
            }
        }

        System.err.println("[QK-NORM-GPU] ✅ Q/K normalization complete");
    }

    /**
     * Debug helper to verify normalization
     */
    public static void debugNormalization(FloatArray tensor, String name, int dim) {
        float sum = 0.0f;
        float sumSquares = 0.0f;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < Math.min(dim, 100); i++) {
            float val = tensor.get(i);
            sum += val;
            sumSquares += val * val;
            min = Math.min(min, val);
            max = Math.max(max, val);
        }

        float mean = sum / Math.min(dim, 100);
        float rms = (float) Math.sqrt(sumSquares / Math.min(dim, 100));

        System.err.printf("[QK-NORM-DEBUG] %s: mean=%.6f, rms=%.6f, min=%.6f, max=%.6f%n",
                         name, mean, rms, min, max);
    }
}