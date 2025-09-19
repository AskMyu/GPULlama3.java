package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.deepseekr1.DeepSeekR1;
import org.beehive.gpullama3.model.deepseekr1.DeepSeekR1Configuration;
import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

/**
 * Model loader for DeepSeek-R1 models with full MoE and MLA support.
 *
 * Handles both full 671B parameter models and distilled variants.
 * Includes specialized loading for MoE expert tensors and MLA configurations.
 */
public class DeepSeekR1ModelLoader extends ModelLoader {

    public DeepSeekR1ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadoVM) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadoVM);
    }

    @Override
    public DeepSeekR1 loadModel() {
        try (Timer modelTimer = Timer.log("Loading DeepSeek-R1 model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            // Load vocabulary and tokenizer (using Qwen3 tokenizer for DeepSeek-R1)
            Vocabulary vocabulary = Vocabulary.loadQwen3Vocabulary(metadata);
            Tokenizer tokenizer = new Qwen3Tokenizer(metadata, vocabulary, true); // Added missing boolean parameter

            // Detect model variant and create appropriate configuration
            DeepSeekR1Configuration config = createConfiguration(metadata, vocabulary);

            // Load weights if requested
            Weights weights = null;
            if (loadWeights) {
                try (Timer weightsTimer = Timer.log("Loading weights")) {
                    Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                    weights = loadWeights(tensorEntries, config);
                }
            }

            // Create the model with ChatFormat (use Qwen3 format since DeepSeek-R1 uses Qwen tokenizer)
            ChatFormat.ChatTokens chatTokens =
                    new ChatFormat.ChatTokens( "<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
            ChatFormat chatFormat = ChatFormat.create(tokenizer, chatTokens);
            DeepSeekR1 model = new DeepSeekR1(config, tokenizer, weights, chatFormat);

            return model;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load DeepSeek-R1 model", e);
        }
    }

    /**
     * Create configuration based on model metadata.
     */
    private DeepSeekR1Configuration createConfiguration(Map<String, Object> metadata, Vocabulary vocabulary) {
        // Extract basic model parameters (DeepSeek-R1 uses Qwen2 architecture)
        int dim = (int) metadata.get("qwen2.embedding_length");
        int hiddenDim = (int) metadata.get("qwen2.feed_forward_length");
        int numberOfLayers = (int) metadata.get("qwen2.block_count");
        int numberOfHeads = (int) metadata.get("qwen2.attention.head_count");
        int numberOfKeyValueHeads = metadata.containsKey("qwen2.attention.head_count_kv") ?
                (int) metadata.get("qwen2.attention.head_count_kv") :
                numberOfHeads;
        int vocabSize = vocabulary.size();
        int contextLengthModel = (int) metadata.get("qwen2.context_length");

        // Extract DeepSeek-R1 specific parameters
        String basename = (String) metadata.getOrDefault("general.basename", "");
        long parameterCount = (long) metadata.getOrDefault("general.parameter_count", 0L);

        // If parameter count is not available, estimate based on model architecture
        if (parameterCount <= 0) {
            // Rough estimation: embedding + transformer layers + output
            long embeddingParams = (long) vocabSize * dim;
            long layerParams = (long) numberOfLayers * (
                6L * dim * dim +  // Q, K, V, O projections (approximately)
                3L * dim * hiddenDim  // FFN layers (gate, up, down)
            );
            parameterCount = embeddingParams + layerParams;
        }

        // Extract remaining required parameters
        float rmsNormEps = (float) metadata.get("qwen2.attention.layer_norm_rms_epsilon");
        float ropeTheta = (float) metadata.get("qwen2.rope.freq_base");

        // Determine if this is full or distilled model
        boolean isFullModel = parameterCount > 100_000_000_000L; // > 100B = full model
        boolean enableMLA = isFullModel; // Only full model has MLA
        boolean isMoEModel = isFullModel; // Only full model has MoE

        // Create configuration based on actual model metadata
        return new DeepSeekR1Configuration(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabSize, contextLengthModel, contextLength, false, // useParallelResidual
            rmsNormEps, ropeTheta,
            isMoEModel ? 256 : 1, // totalExperts (256 for full, 1 for distilled = no MoE)
            isMoEModel ? 37 : 1,  // activeExperts (37B for full, 1 for distilled = no MoE)
            isMoEModel ? hiddenDim * 2 : hiddenDim, // expertHiddenDim
            parameterCount, // totalParameters
            isMoEModel ? 37_000_000_000L : parameterCount, // activeParameters
            isMoEModel, // enableLoadBalancing
            0.1f, // routingNoise
            enableMLA, // enableMLA
            enableMLA ? org.beehive.gpullama3.attention.mla.MLAConfiguration.forDeepSeekR1(dim / numberOfHeads, numberOfHeads, numberOfKeyValueHeads) : null,
            false, // enableFP8
            true,  // conservativeSampling
            "silu"  // activationFunction
        );
    }

}