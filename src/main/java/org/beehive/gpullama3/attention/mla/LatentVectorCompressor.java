package org.beehive.gpullama3.attention.mla;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Compresses Key and Value matrices into latent vectors for Multi-head Latent Attention.
 *
 * This is the core innovation of DeepSeek-R1's MLA mechanism, achieving 85-95% memory
 * reduction in KV cache while maintaining attention quality.
 *
 * ISOLATION: This class operates independently and does not affect existing attention mechanisms.
 */
public class LatentVectorCompressor {

    private final MLAConfiguration config;
    private final FloatArray compressionMatrix;  // Learned compression transformation
    private final FloatArray compressionBias;    // Optional bias for compression

    public LatentVectorCompressor(MLAConfiguration config) {
        this.config = config;

        // Initialize compression matrix: [headDim, latentDim]
        this.compressionMatrix = new FloatArray(config.headDim() * config.latentDim());
        this.compressionBias = new FloatArray(config.latentDim());

        initializeCompressionParameters();
    }

    /**
     * Compress Key matrix into latent vectors.
     *
     * @param keys Input key matrix [batchSize * seqLen, numHeads * headDim]
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @return Compressed latent keys [batchSize * seqLen, kvHeads * latentDim]
     */
    public FloatArray compressKeys(FloatArray keys, int batchSize, int seqLen) {
        int inputSize = batchSize * seqLen * config.kvHeads() * config.headDim();
        int outputSize = batchSize * seqLen * config.kvHeads() * config.latentDim();

        if (keys.getSize() != inputSize) {
            throw new IllegalArgumentException("Key matrix size mismatch: expected " + inputSize + ", got " + keys.getSize());
        }

        FloatArray compressedKeys = new FloatArray(outputSize);
        performCompression(keys, compressedKeys, batchSize, seqLen, true);

        return compressedKeys;
    }

    /**
     * Compress Value matrix into latent vectors.
     *
     * @param values Input value matrix [batchSize * seqLen, numHeads * headDim]
     * @param batchSize Batch size
     * @param seqLen Sequence length
     * @return Compressed latent values [batchSize * seqLen, kvHeads * latentDim]
     */
    public FloatArray compressValues(FloatArray values, int batchSize, int seqLen) {
        int inputSize = batchSize * seqLen * config.kvHeads() * config.headDim();
        int outputSize = batchSize * seqLen * config.kvHeads() * config.latentDim();

        if (values.getSize() != inputSize) {
            throw new IllegalArgumentException("Value matrix size mismatch: expected " + inputSize + ", got " + values.getSize());
        }

        FloatArray compressedValues = new FloatArray(outputSize);
        performCompression(values, compressedValues, batchSize, seqLen, false);

        return compressedValues;
    }

    /**
     * Core compression operation using learned linear transformation.
     *
     * Applies: output = input * compressionMatrix + compressionBias
     */
    private void performCompression(FloatArray input, FloatArray output, int batchSize, int seqLen, boolean isKeys) {
        int headDim = config.headDim();
        int latentDim = config.latentDim();
        int kvHeads = config.kvHeads();

        for (int batch = 0; batch < batchSize; batch++) {
            for (int seq = 0; seq < seqLen; seq++) {
                for (int head = 0; head < kvHeads; head++) {
                    // Input vector for this (batch, seq, head)
                    int inputBaseIdx = ((batch * seqLen + seq) * kvHeads + head) * headDim;

                    // Output vector for this (batch, seq, head)
                    int outputBaseIdx = ((batch * seqLen + seq) * kvHeads + head) * latentDim;

                    // Matrix multiplication: input[headDim] * compressionMatrix[headDim, latentDim] = output[latentDim]
                    for (int latent = 0; latent < latentDim; latent++) {
                        float sum = 0.0f;

                        // Dot product with compression matrix column
                        for (int dim = 0; dim < headDim; dim++) {
                            float inputVal = input.get(inputBaseIdx + dim);
                            float weight = compressionMatrix.get(dim * latentDim + latent);
                            sum += inputVal * weight;
                        }

                        // Add bias and store result
                        sum += compressionBias.get(latent);
                        output.set(outputBaseIdx + latent, sum);
                    }
                }
            }
        }
    }

    /**
     * Initialize compression parameters with small random weights.
     * In a real implementation, these would be loaded from trained model weights.
     */
    private void initializeCompressionParameters() {
        // Xavier/Glorot initialization for compression matrix
        float scale = (float) Math.sqrt(2.0 / (config.headDim() + config.latentDim()));

        for (int i = 0; i < compressionMatrix.getSize(); i++) {
            // Small random initialization
            float value = (float) (Math.random() * 2 * scale - scale);
            compressionMatrix.set(i, value);
        }

        // Initialize bias to zero
        for (int i = 0; i < compressionBias.getSize(); i++) {
            compressionBias.set(i, 0.0f);
        }
    }

    /**
     * Set compression parameters from trained model weights.
     */
    public void setCompressionWeights(FloatArray weights, FloatArray bias) {
        if (weights.getSize() != compressionMatrix.getSize()) {
            throw new IllegalArgumentException("Compression weights size mismatch");
        }
        if (bias.getSize() != compressionBias.getSize()) {
            throw new IllegalArgumentException("Compression bias size mismatch");
        }

        // Copy weights
        for (int i = 0; i < weights.getSize(); i++) {
            compressionMatrix.set(i, weights.get(i));
        }

        // Copy bias
        for (int i = 0; i < bias.getSize(); i++) {
            compressionBias.set(i, bias.get(i));
        }
    }

    /**
     * Get memory usage of this compressor.
     */
    public long getMemoryUsage() {
        return (compressionMatrix.getSize() + compressionBias.getSize()) * 4L; // 4 bytes per float
    }

    /**
     * Calculate compression ratio achieved.
     */
    public float getCompressionRatio() {
        return config.compressionRatio();
    }

    /**
     * Get the MLA configuration.
     */
    public MLAConfiguration getConfig() {
        return config;
    }
}