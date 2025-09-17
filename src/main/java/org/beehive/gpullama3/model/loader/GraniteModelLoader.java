package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.GraniteChatFormat;
import org.beehive.gpullama3.model.granite.Granite;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tokenizer.impl.GraniteTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

/**
 * Model loader for IBM Granite 3.3 models.
 * Handles the specific GGUF format and metadata for Granite models.
 */
public class GraniteModelLoader extends ModelLoader {

    public GraniteModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    public Model loadModel() {
        Map<String, Object> metadata = gguf.getMetadata();

        // Extract Granite-specific configuration from metadata
        int dim = (int) metadata.getOrDefault("granite.embedding_length", 
                     metadata.getOrDefault("llama.embedding_length", 2048));
        
        int hiddenDim = (int) metadata.getOrDefault("granite.feed_forward_length",
                          metadata.getOrDefault("llama.feed_forward_length", 5504));
        
        int numberOfLayers = (int) metadata.getOrDefault("granite.block_count",
                               metadata.getOrDefault("llama.block_count", 24));
        
        int numberOfHeads = (int) metadata.getOrDefault("granite.attention.head_count",
                              metadata.getOrDefault("llama.attention.head_count", 16));
        
        // Granite uses GQA (Grouped Query Attention)
        int numberOfKeyValueHeads = (int) metadata.getOrDefault("granite.attention.head_count_kv",
                                       metadata.getOrDefault("llama.attention.head_count_kv", 4));
        
        // Granite vocabulary size - Granite models typically have ~49159 vocabulary
        int vocabularySize = (int) metadata.getOrDefault("granite.vocab_size",
                               metadata.getOrDefault("llama.vocab_size", 49159));
        
        // Context length with fallback
        int contextLengthConfig = (int) metadata.getOrDefault("granite.context_length",
                                     metadata.getOrDefault("llama.context_length", 131072));
        
        // Use provided context length if valid, otherwise use config
        if (contextLength <= 0) {
            contextLength = contextLengthConfig;
        }
        
        float rmsNormEps = ((Number) metadata.getOrDefault("granite.attention.layer_norm_rms_epsilon",
                             metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f))).floatValue();
        
        float ropeTheta = ((Number) metadata.getOrDefault("granite.rope.freq_base",
                            metadata.getOrDefault("llama.rope.freq_base", 10000.0f))).floatValue();

        // Create Granite configuration
        GraniteConfiguration config = new GraniteConfiguration(
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
        // Granite uses a similar vocabulary structure to other models
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        Vocabulary vocabulary = new Vocabulary(tokens, scores);
        Tokenizer tokenizer = new GraniteTokenizer(vocabulary);

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
                throw new RuntimeException("Failed to load Granite model weights", e);
            }
        }

        // Create chat format handler
        ChatFormat chatFormat = new GraniteChatFormat(tokenizer);

        // Return the loaded Granite model
        return new Granite(config, tokenizer, weights, chatFormat);
    }
}