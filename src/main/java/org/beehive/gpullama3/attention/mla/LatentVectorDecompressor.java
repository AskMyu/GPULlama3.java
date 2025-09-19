package org.beehive.gpullama3.attention.mla;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Decompresses latent vectors back to Key and Value matrices for attention computation.
 *
 * This is performed on-demand during attention computation, allowing MLA to maintain
 * compressed storage while providing full-dimensional vectors for attention operations.
 *
 * ISOLATION: This class operates independently and does not affect existing attention mechanisms.
 */
public class LatentVectorDecompressor {

    private final MLAConfiguration config;
    private final FloatArray decompressionMatrix;  // Learned decompression transformation
    private final FloatArray decompressionBias;    // Optional bias for decompression

    public LatentVectorDecompressor(MLAConfiguration config) {
        this.config = config;

        // Initialize decompression matrix: [latentDim, headDim]
        this.decompressionMatrix = new FloatArray(config.latentDim() * config.headDim());
        this.decompressionBias = new FloatArray(config.headDim());

        initializeDecompressionParameters();
    }

    /**
     * Decompress latent keys back to full Key matrix for attention computation.
     *
     * @param latentKeys Compressed latent keys [batchSize * seqLen, kvHeads * latentDim]
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @return Decompressed keys [batchSize * seqLen, kvHeads * headDim]
     */
    public FloatArray decompressKeys(FloatArray latentKeys, int batchSize, int seqLen) {
        int inputSize = batchSize * seqLen * config.kvHeads() * config.latentDim();
        int outputSize = batchSize * seqLen * config.kvHeads() * config.headDim();

        if (latentKeys.getSize() != inputSize) {
            throw new IllegalArgumentException("Latent keys size mismatch: expected " + inputSize + ", got " + latentKeys.getSize());
        }

        FloatArray decompressedKeys = new FloatArray(outputSize);
        performDecompression(latentKeys, decompressedKeys, batchSize, seqLen, true);

        return decompressedKeys;
    }

    /**
     * Decompress latent values back to full Value matrix for attention computation.
     *
     * @param latentValues Compressed latent values [batchSize * seqLen, kvHeads * latentDim]
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @return Decompressed values [batchSize * seqLen, kvHeads * headDim]
     */
    public FloatArray decompressValues(FloatArray latentValues, int batchSize, int seqLen) {
        int inputSize = batchSize * seqLen * config.kvHeads() * config.latentDim();
        int outputSize = batchSize * seqLen * config.kvHeads() * config.headDim();

        if (latentValues.getSize() != inputSize) {
            throw new IllegalArgumentException("Latent values size mismatch: expected " + inputSize + ", got " + latentValues.getSize());
        }

        FloatArray decompressedValues = new FloatArray(outputSize);
        performDecompression(latentValues, decompressedValues, batchSize, seqLen, false);

        return decompressedValues;
    }

    /**
     * Selective decompression for specific attention heads only.
     * This optimization allows processing only the heads needed for current computation.
     *
     * @param latentVectors Input latent vectors
     * @param targetHeads Array of head indices to decompress
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @param isKeys Whether decompressing keys (true) or values (false)
     * @return Decompressed vectors for target heads only
     */
    public FloatArray decompressSelectiveHeads(FloatArray latentVectors, int[] targetHeads,
                                             int batchSize, int seqLen, boolean isKeys) {
        int numTargetHeads = targetHeads.length;
        int outputSize = batchSize * seqLen * numTargetHeads * config.headDim();
        FloatArray selectiveOutput = new FloatArray(outputSize);

        int latentDim = config.latentDim();
        int headDim = config.headDim();
        int kvHeads = config.kvHeads();

        for (int batch = 0; batch < batchSize; batch++) {
            for (int seq = 0; seq < seqLen; seq++) {
                for (int targetIdx = 0; targetIdx < numTargetHeads; targetIdx++) {
                    int headIdx = targetHeads[targetIdx];

                    if (headIdx >= kvHeads) {
                        throw new IllegalArgumentException("Head index " + headIdx + " exceeds available KV heads " + kvHeads);
                    }

                    // Input latent vector for this (batch, seq, head)
                    int inputBaseIdx = ((batch * seqLen + seq) * kvHeads + headIdx) * latentDim;

                    // Output vector for this (batch, seq, targetHead)
                    int outputBaseIdx = ((batch * seqLen + seq) * numTargetHeads + targetIdx) * headDim;

                    // Matrix multiplication: latent[latentDim] * decompressionMatrix[latentDim, headDim] = output[headDim]
                    for (int dim = 0; dim < headDim; dim++) {
                        float sum = 0.0f;

                        // Dot product with decompression matrix row
                        for (int latent = 0; latent < latentDim; latent++) {
                            float latentVal = latentVectors.get(inputBaseIdx + latent);
                            float weight = decompressionMatrix.get(latent * headDim + dim);
                            sum += latentVal * weight;
                        }

                        // Add bias and store result
                        sum += decompressionBias.get(dim);
                        selectiveOutput.set(outputBaseIdx + dim, sum);
                    }
                }
            }
        }

        return selectiveOutput;
    }

    /**
     * Core decompression operation using learned linear transformation.
     *
     * Applies: output = latent * decompressionMatrix + decompressionBias
     */
    private void performDecompression(FloatArray latent, FloatArray output, int batchSize, int seqLen, boolean isKeys) {
        int latentDim = config.latentDim();
        int headDim = config.headDim();
        int kvHeads = config.kvHeads();

        for (int batch = 0; batch < batchSize; batch++) {
            for (int seq = 0; seq < seqLen; seq++) {
                for (int head = 0; head < kvHeads; head++) {
                    // Input latent vector for this (batch, seq, head)
                    int inputBaseIdx = ((batch * seqLen + seq) * kvHeads + head) * latentDim;

                    // Output vector for this (batch, seq, head)
                    int outputBaseIdx = ((batch * seqLen + seq) * kvHeads + head) * headDim;

                    // Matrix multiplication: latent[latentDim] * decompressionMatrix[latentDim, headDim] = output[headDim]
                    for (int dim = 0; dim < headDim; dim++) {
                        float sum = 0.0f;

                        // Dot product with decompression matrix row
                        for (int l = 0; l < latentDim; l++) {
                            float latentVal = latent.get(inputBaseIdx + l);
                            float weight = decompressionMatrix.get(l * headDim + dim);
                            sum += latentVal * weight;
                        }

                        // Add bias and store result
                        sum += decompressionBias.get(dim);
                        output.set(outputBaseIdx + dim, sum);
                    }
                }
            }
        }
    }

    /**
     * Initialize decompression parameters with small random weights.
     * In a real implementation, these would be loaded from trained model weights.
     */
    private void initializeDecompressionParameters() {
        // Xavier/Glorot initialization for decompression matrix
        float scale = (float) Math.sqrt(2.0 / (config.latentDim() + config.headDim()));

        for (int i = 0; i < decompressionMatrix.getSize(); i++) {
            // Small random initialization
            float value = (float) (Math.random() * 2 * scale - scale);
            decompressionMatrix.set(i, value);
        }

        // Initialize bias to zero
        for (int i = 0; i < decompressionBias.getSize(); i++) {
            decompressionBias.set(i, 0.0f);
        }
    }

    /**
     * Set decompression parameters from trained model weights.
     */
    public void setDecompressionWeights(FloatArray weights, FloatArray bias) {
        if (weights.getSize() != decompressionMatrix.getSize()) {
            throw new IllegalArgumentException("Decompression weights size mismatch");
        }
        if (bias.getSize() != decompressionBias.getSize()) {
            throw new IllegalArgumentException("Decompression bias size mismatch");
        }

        // Copy weights
        for (int i = 0; i < weights.getSize(); i++) {
            decompressionMatrix.set(i, weights.get(i));
        }

        // Copy bias
        for (int i = 0; i < bias.getSize(); i++) {
            decompressionBias.set(i, bias.get(i));
        }
    }

    /**
     * Get memory usage of this decompressor.
     */
    public long getMemoryUsage() {
        return (decompressionMatrix.getSize() + decompressionBias.getSize()) * 4L; // 4 bytes per float
    }

    /**
     * Get expansion ratio (inverse of compression).
     */
    public float getExpansionRatio() {
        return 1.0f / config.compressionRatio();
    }

    /**
     * Get the MLA configuration.
     */
    public MLAConfiguration getConfig() {
        return config;
    }
}