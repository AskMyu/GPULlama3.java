package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.GemmaChatFormat;
import org.beehive.gpullama3.model.gemma.Gemma;
import org.beehive.gpullama3.model.gemma.GemmaConfiguration;
import org.beehive.gpullama3.tokenizer.impl.GemmaTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

/**
 * Model loader for Gemma 3 models.
 * Handles the specific GGUF format and metadata for Gemma models.
 */
public class GemmaModelLoader extends ModelLoader {

    public GemmaModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    public Model loadModel() {
        Map<String, Object> metadata = gguf.getMetadata();

        // Debug: Print all available metadata keys
        System.out.println("DEBUG: Available metadata keys:");
        metadata.keySet().stream().sorted().forEach(key -> {
            Object value = metadata.get(key);
            if (key.contains("embed") || key.contains("block") || key.contains("head") || key.contains("vocab") || key.contains("context") || key.contains("layer") || key.contains("token")) {
                System.out.println("  " + key + " = " + value + " (" + (value != null ? value.getClass().getSimpleName() : "null") + ")");
            }
        });

        // Extract Gemma-specific configuration from metadata using correct gemma3. prefix
        int dim = (int) metadata.getOrDefault("gemma3.embedding_length", 
                     metadata.getOrDefault("gemma.embedding_length", 
                     metadata.getOrDefault("llama.embedding_length", 640)));
        
        int hiddenDim = (int) metadata.getOrDefault("gemma3.feed_forward_length",
                          metadata.getOrDefault("gemma.feed_forward_length",
                          metadata.getOrDefault("llama.feed_forward_length", dim * 4))); // Standard 4x multiplier
        
        int numberOfLayers = (int) metadata.getOrDefault("gemma3.block_count",
                               metadata.getOrDefault("gemma.block_count",
                               metadata.getOrDefault("llama.block_count", 18)));
        
        int numberOfHeads = (int) metadata.getOrDefault("gemma3.attention.head_count",
                              metadata.getOrDefault("gemma.attention.head_count",
                              metadata.getOrDefault("llama.attention.head_count", 4)));
        
        int numberOfKeyValueHeads = (int) metadata.getOrDefault("gemma3.attention.head_count_kv",
                                       metadata.getOrDefault("gemma.attention.head_count_kv",
                                       metadata.getOrDefault("llama.attention.head_count_kv", 1)));
        
        // Gemma's vocabulary size - get from tokenizer tokens array length
        int vocabularySize;
        if (metadata.containsKey("tokenizer.ggml.tokens")) {
            String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
            vocabularySize = tokens.length;
        } else {
            vocabularySize = (int) metadata.getOrDefault("gemma3.vocab_size",
                               metadata.getOrDefault("gemma.vocab_size",
                               metadata.getOrDefault("llama.vocab_size", 256000)));
        }
        
        // Context length with fallback
        int contextLengthConfig = (int) metadata.getOrDefault("gemma3.context_length",
                                     metadata.getOrDefault("gemma.context_length",
                                     metadata.getOrDefault("llama.context_length", 32768)));
        
        // Use provided context length if valid, otherwise use config
        if (contextLength <= 0) {
            contextLength = contextLengthConfig;
        }
        
        float rmsNormEps = ((Number) metadata.getOrDefault("gemma3.attention.layer_norm_rms_epsilon",
                             metadata.getOrDefault("gemma.attention.layer_norm_rms_epsilon",
                             metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-6f)))).floatValue();
        
        float ropeTheta = ((Number) metadata.getOrDefault("gemma3.rope.freq_base",
                            metadata.getOrDefault("gemma.rope.freq_base",
                            metadata.getOrDefault("llama.rope.freq_base", 10000.0f)))).floatValue();
        
        // Debug: Print the configuration we're using
        System.out.println("DEBUG: Gemma configuration:");
        System.out.println("  dim: " + dim);
        System.out.println("  hiddenDim: " + hiddenDim);  
        System.out.println("  numberOfLayers: " + numberOfLayers);
        System.out.println("  numberOfHeads: " + numberOfHeads);
        System.out.println("  numberOfKeyValueHeads: " + numberOfKeyValueHeads);
        System.out.println("  vocabularySize: " + vocabularySize);
        System.out.println("  contextLength: " + contextLength);

        // Create Gemma configuration
        GemmaConfiguration config = new GemmaConfiguration(
            dim,
            hiddenDim,
            numberOfLayers,
            numberOfHeads,
            numberOfKeyValueHeads,
            vocabularySize,
            contextLength,
            rmsNormEps,
            ropeTheta
        );

        // Load vocabulary and create tokenizer
        // Gemma uses a similar vocabulary structure to Llama
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        Vocabulary vocabulary = new Vocabulary(tokens, scores);
        Tokenizer tokenizer = new GemmaTokenizer(vocabulary);

        // INFO: Check token embedding size for memory planning
        long estimatedTokenEmbeddingSize = (long) vocabularySize * dim;
        System.err.printf("[GEMMA-INFO] Token embedding tensor: %d × %d = %d elements (%.2f GB)%n",
                          vocabularySize, dim, estimatedTokenEmbeddingSize, estimatedTokenEmbeddingSize * 4.0 / (1024*1024*1024));

        if (estimatedTokenEmbeddingSize > Integer.MAX_VALUE) {
            System.err.printf("[GEMMA-WARNING] Large tensor detected - may hit allocation limits%n");
        }

        // Load weights if requested
        Weights weights = null;
        if (loadWeights) {
            try {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(
                    fileChannel,
                    gguf.getTensorDataOffset(),
                    gguf.getTensorInfos()
                );
                weights = loadWeights(tensorEntries, config);
            } catch (IOException e) {
                throw new RuntimeException("Failed to load Gemma model weights", e);
            }
        }

        // Create chat format handler
        ChatFormat chatFormat = new GemmaChatFormat(tokenizer);

        // Return the loaded Gemma model
        return new Gemma(config, tokenizer, weights, chatFormat);
    }
}