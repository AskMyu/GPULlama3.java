package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Phi3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.phi4.Phi4;
import org.beehive.gpullama3.model.phi4.Phi4Configuration;
import org.beehive.gpullama3.tokenizer.impl.Phi3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadPhi3Vocabulary;

public class Phi4ModelLoader extends ModelLoader {

    public Phi4ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    // @formatter:off
    @Override
    public Phi4 loadModel() {
        try (var ignored = Timer.log("Load Phi4 model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            // Load vocabulary - Phi-4 uses a much larger vocabulary than Phi-3
            Vocabulary vocabulary = loadPhi3Vocabulary(metadata);
            boolean isPhi4MiniReasoning = isPhi4ReasoningModel(metadata);
            
            // Use Phi3Tokenizer as base, but with Phi-4 specific configurations
            Tokenizer tokenizer = new Phi3Tokenizer(metadata, vocabulary);

            // Phi-4 models use phi3 architecture, so look for phi3 metadata keys
            Object contextLengthObj = metadata.get("phi3.context_length");
            if (contextLengthObj == null) {
                contextLengthObj = metadata.get("llama.context_length");
            }
            if (contextLengthObj != null) {
                int contextLength = (int) contextLengthObj;
                if (contextLength < 0 || contextLength < contextLength) {
                    contextLength = contextLength;
                }
            }

            Phi4Configuration config;
            if (isPhi4MiniReasoning) {
                // Use phi3 metadata keys since architecture is phi3
                config = Phi4Configuration.createPhi4MiniReasoning(
                        (int) metadata.get("phi3.embedding_length"),
                        (int) metadata.get("phi3.feed_forward_length"),
                        (int) metadata.get("phi3.block_count"),
                        (int) metadata.get("phi3.attention.head_count"),
                        metadata.containsKey("phi3.attention.head_count_kv")
                                ? (int) metadata.get("phi3.attention.head_count_kv")
                                : (int) metadata.get("phi3.attention.head_count"),
                        vocabulary.size(),
                        contextLength,
                        (float) metadata.get("phi3.attention.layer_norm_rms_epsilon"),
                        (float) metadata.get("phi3.rope.freq_base")
                );
            } else {
                config = Phi4Configuration.createPhi4(
                        (int) metadata.get("phi3.embedding_length"),
                        (int) metadata.get("phi3.feed_forward_length"),
                        (int) metadata.get("phi3.block_count"),
                        (int) metadata.get("phi3.attention.head_count"),
                        metadata.containsKey("phi3.attention.head_count_kv")
                                ? (int) metadata.get("phi3.attention.head_count_kv")
                                : (int) metadata.get("phi3.attention.head_count"),
                        vocabulary.size(),
                        contextLength,
                        (float) metadata.get("phi3.attention.layer_norm_rms_epsilon"),
                        (float) metadata.get("phi3.rope.freq_base")
                );
            }

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }

            // Phi-4-Mini-Reasoning uses specific chat tokens for reasoning
            ChatTokens chatTokens = isPhi4MiniReasoning ?
                    new ChatTokens("<|user|>", "<|end|>", "<|assistant|>", "<|endoftext|>", "<|reasoning|>") :
                    new ChatTokens("<|user|>", "<|end|>", "<|assistant|>", "<|endoftext|>", "");
            
            return new Phi4(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLength(),
                config.headSize(),
                config.ropeTheta(),
                false,
                0,
                0,
                0,
                0
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        // Phi-4 uses shared embeddings
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault("output.weight", tokenEmbeddings);

        if (Options.getDefaultOptions().useTornadovm()) {
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                System.out.println("Loading Phi4 model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
            }
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, 
            Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        // Reuse Phi3TornadoWeights as the architecture is similar
        return new Phi3TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                loadTensorAsHalfFloatArray(outputWeight),
                outputWeight.ggmlType()
        );
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                         Configuration config,
                                         Pair<float[], float[]> ropeFreqs,
                                         GGMLTensorEntry tokenEmbeddings,
                                         GGMLTensorEntry outputWeight) {
        // Reuse Phi3StandardWeights as the architecture is similar
        return new Phi3StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType()
        );
    }
    // @formatter:on
    
    /**
     * Checks if this is a Phi-4-Mini-Reasoning model
     */
    private boolean isPhi4ReasoningModel(Map<String, Object> metadata) {
        String modelName = (String) metadata.get("general.name");
        String architecture = (String) metadata.get("general.architecture");
        
        if (modelName != null) {
            String lowerName = modelName.toLowerCase();
            if (lowerName.contains("phi-4-mini-reasoning") || 
                lowerName.contains("phi4-mini-reasoning") ||
                lowerName.contains("phi-4 mini reasoning")) {
                return true;
            }
        }
        
        // Check architecture type
        return "phi4".equals(architecture) || "phi-4".equals(architecture);
    }
}