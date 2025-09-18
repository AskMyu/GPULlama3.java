package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.GemmaChatFormat;
import org.beehive.gpullama3.model.gemma.Gemma2;
import org.beehive.gpullama3.model.gemma.Gemma2Configuration;
import org.beehive.gpullama3.tokenizer.impl.GemmaTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

/**
 * Model loader specifically for Gemma 2 models.
 * Handles Gemma 2-specific configuration including soft-capping parameters.
 * SEPARATE from GemmaModelLoader to maintain clean architecture separation.
 */
public class Gemma2ModelLoader extends ModelLoader {

    public Gemma2ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    public Model loadModel() {
        Map<String, Object> metadata = gguf.getMetadata();

        // Extract Gemma 2-specific configuration
        // Gemma 2 2B standard parameters (based on research)
        int dim = (int) metadata.getOrDefault("gemma.embedding_length", 2304);
        int hiddenDim = (int) metadata.getOrDefault("gemma.feed_forward_length", 9216);
        int numberOfLayers = (int) metadata.getOrDefault("gemma.block_count", 26);
        int numberOfHeads = (int) metadata.getOrDefault("gemma.attention.head_count", 8);
        int numberOfKeyValueHeads = (int) metadata.getOrDefault("gemma.attention.head_count_kv", 4);

        // Vocabulary size from tokenizer
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        int vocabularySize = tokens.length;

        // TEMPORARY: Limit vocabulary size to stay under OpenCL memory limits
        // Embedding: vocabSize * dim * 4 bytes must be < 1.885GB
        // Max vocab = 1.885GB / (2304 * 4) = ~216,000 tokens
        int maxVocabForGPU = (int) (1.885 * 1024 * 1024 * 1024 / (dim * 4));
        if (vocabularySize > maxVocabForGPU) {
            System.err.printf("[GEMMA2-LOADER] TEMP: Reducing vocabulary from %d to %d for GPU memory limits%n",
                            vocabularySize, maxVocabForGPU);
            vocabularySize = maxVocabForGPU;
        }

        // Gemma 2 context length is 8192 (NOT 32K like Gemma 3)
        int contextLengthConfig = (int) metadata.getOrDefault("gemma.context_length", 8192);
        if (contextLength <= 0) {
            contextLength = contextLengthConfig;
        }

        // TEMPORARY: Reduce context length for OpenCL memory limit testing
        if (contextLength > 2048) {
            System.err.printf("[GEMMA2-LOADER] TEMP: Reducing context length from %d to 2048 for GPU memory testing%n", contextLength);
            contextLength = 2048;
        }

        float rmsNormEps = ((Number) metadata.getOrDefault("gemma.attention.layer_norm_rms_epsilon", 1e-6f)).floatValue();
        float ropeTheta = ((Number) metadata.getOrDefault("gemma.rope.freq_base", 10000.0f)).floatValue();

        // Gemma 2 soft-capping parameters (CRITICAL for oscillation fix)
        float finalLogitSoftcapping = 30.0f;    // Research-based value
        float attnLogitSoftcapping = 50.0f;     // Research-based value

        System.out.println("DEBUG: Gemma 2 configuration:");
        System.out.println("  dim: " + dim + " (should be 2304 for 2B)");
        System.out.println("  numberOfLayers: " + numberOfLayers + " (should be 26 for 2B)");
        System.out.println("  numberOfHeads: " + numberOfHeads + " (should be 8 for 2B)");
        System.out.println("  numberOfKeyValueHeads: " + numberOfKeyValueHeads + " (should be 4 for 2B)");
        System.out.println("  contextLength: " + contextLength + " (should be 8192 for Gemma 2)");
        System.out.println("  finalLogitSoftcapping: " + finalLogitSoftcapping);
        System.out.println("  attnLogitSoftcapping: " + attnLogitSoftcapping);

        // Create Gemma 2-specific configuration
        Gemma2Configuration config = new Gemma2Configuration(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabularySize, contextLength, rmsNormEps, ropeTheta,
            finalLogitSoftcapping, attnLogitSoftcapping
        );

        // Use existing GemmaTokenizer (compatible with Gemma 2)
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        Vocabulary vocabulary = new Vocabulary(tokens, scores);
        Tokenizer tokenizer = new GemmaTokenizer(vocabulary);

        // Load weights using existing infrastructure
        Weights weights = null;
        if (loadWeights) {
            try {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(
                    fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos()
                );
                weights = loadWeights(tensorEntries, config);
            } catch (IOException e) {
                throw new RuntimeException("Failed to load Gemma 2 model weights", e);
            }
        }

        // Use existing GemmaChatFormat (compatible with Gemma 2)
        ChatFormat chatFormat = new GemmaChatFormat(tokenizer);

        return new Gemma2(config, tokenizer, weights, chatFormat);
    }
}