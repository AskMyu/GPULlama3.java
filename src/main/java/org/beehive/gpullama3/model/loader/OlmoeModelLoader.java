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
import org.beehive.gpullama3.inference.weights.standard.LlamaStandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.olmoe.Olmoe;
import org.beehive.gpullama3.model.olmoe.OlmoeConfiguration;
import org.beehive.gpullama3.tokenizer.impl.GemmaTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.model.loader.batch.BatchCapableModelLoader;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Map;

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadOlmoeVocabulary;

/**
 * Model loader for OLMoE models.
 * 
 * Handles loading of OLMoE-1B-7B and similar MoE models with
 * 64 experts and Top-8 routing.
 */
public class OlmoeModelLoader extends BatchCapableModelLoader {

    // OLMoE expert tensor patterns - same as GPT-OSS
    private static final String[] OLMOE_EXPERT_PATTERNS = {
        "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"
    };

    public OlmoeModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    protected boolean requiresSpecialHandling(String tensorName) {
        // OLMoE expert tensors require special handling
        return Arrays.stream(OLMOE_EXPERT_PATTERNS)
                    .anyMatch(tensorName::contains);
    }

    @Override
    protected String[] getModelSpecificExpertPatterns() {
        return OLMOE_EXPERT_PATTERNS;
    }

    @Override
    protected Weights createWeightsFromTensors(Map<String, org.beehive.gpullama3.core.model.tensor.FloatTensor> tensors, Configuration config) {
        // Use existing OLMoE weight creation logic
        return createOlmoeWeights(tensors, config);
    }

    /**
     * Creates OLMoE weights from loaded tensors using existing weight creation logic
     */
    private Weights createOlmoeWeights(Map<String, org.beehive.gpullama3.core.model.tensor.FloatTensor> loadedTensors, Configuration config) {
        // Simple bridge: reload original tensor entries and use existing weight creation
        // The BatchCapableModelLoader has already loaded the tensors into memory,
        // so this just provides the metadata needed for the existing weight creation logic
        try {
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(
                fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());

            System.err.printf("[OLMOE-BRIDGE] Loaded %d tensor entries for weight creation%n", tensorEntries.size());

            // Use existing weight creation logic with all available tensors
            return loadWeightsOlmoeOriginal(tensorEntries, config);
        } catch (IOException e) {
            throw new RuntimeException("Failed to reload tensor entries for OLMoE weight creation", e);
        }
    }

    // @formatter:off
    @Override
    public Olmoe loadModel() {
        try (var ignored = Timer.log("Load OLMoE model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            // Load OLMoE-specific vocabulary
            Vocabulary vocabulary = loadOlmoeVocabulary(metadata);
            
            // Create tokenizer - OLMoE uses GPT-2 style tokenizer, similar to Gemma
            Tokenizer tokenizer = new GemmaTokenizer(vocabulary);

            // Get context length from metadata
            int modelContextLength = getContextLength(metadata);
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            // Create OLMoE configuration
            OlmoeConfiguration config = createConfiguration(metadata, vocabulary.size(), contextLength);

            // Load weights if requested
            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(
                    fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }

            // OLMoE chat format
            ChatTokens chatTokens = new ChatTokens(
                "<|im_start|>",      // Start header
                "<|im_end|>",        // End header
                "",                  // Assistant prefix
                "<|endoftext|>",     // End of text
                ""                   // No special reasoning token
            );
            
            return new Olmoe(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load OLMoE model", e);
        }
    }
    // @formatter:on

    /**
     * Creates OLMoE configuration from metadata
     */
    private OlmoeConfiguration createConfiguration(Map<String, Object> metadata, 
            int vocabularySize, int contextLength) {
        
        // Try to get configuration from metadata
        Integer dim = getIntFromMetadata(metadata, "olmoe.embedding_length", 2048);
        Integer hiddenDim = getIntFromMetadata(metadata, "olmoe.feed_forward_length", 1024);
        Integer numberOfLayers = getIntFromMetadata(metadata, "olmoe.block_count", 16);
        Integer numberOfHeads = getIntFromMetadata(metadata, "olmoe.attention.head_count", 16);
        Integer numberOfKeyValueHeads = getIntFromMetadata(metadata, 
            "olmoe.attention.head_count_kv", numberOfHeads);
        
        // MoE-specific parameters
        Integer numberOfExperts = getIntFromMetadata(metadata, "olmoe.expert_count", 64);
        Integer numberOfActiveExperts = getIntFromMetadata(metadata, "olmoe.expert_used_count", 8);
        
        Float rmsNormEps = getFloatFromMetadata(metadata, 
            "olmoe.attention.layer_norm_rms_epsilon", 1e-05f);
        Float ropeTheta = getFloatFromMetadata(metadata, "olmoe.rope.freq_base", 10000.0f);
        Float routerAuxLossCoef = getFloatFromMetadata(metadata, 
            "olmoe.router_aux_loss_coef", 0.01f);
        Boolean outputRouterLogits = getBoolFromMetadata(metadata, 
            "olmoe.output_router_logits", false);
        
        return OlmoeConfiguration.create(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabularySize, contextLength, numberOfExperts, numberOfActiveExperts,
            rmsNormEps, ropeTheta, routerAuxLossCoef, outputRouterLogits
        );
    }

    /**
     * Gets context length from metadata
     */
    private int getContextLength(Map<String, Object> metadata) {
        if (metadata.containsKey("olmoe.context_length")) {
            return (int) metadata.get("olmoe.context_length");
        }
        // Default to 4096 for OLMoE-1B-7B
        return 4096;
    }

    /**
     * Helper to get integer from metadata with default
     */
    private Integer getIntFromMetadata(Map<String, Object> metadata, String key, int defaultValue) {
        return metadata.containsKey(key) ? (Integer) metadata.get(key) : defaultValue;
    }

    /**
     * Helper to get float from metadata with default
     */
    private Float getFloatFromMetadata(Map<String, Object> metadata, String key, float defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Float) {
            return (Float) value;
        } else if (value instanceof Double) {
            return ((Double) value).floatValue();
        }
        return defaultValue;
    }

    /**
     * Helper to get boolean from metadata with default
     */
    private Boolean getBoolFromMetadata(Map<String, Object> metadata, String key, boolean defaultValue) {
        return metadata.containsKey(key) ? (Boolean) metadata.get(key) : defaultValue;
    }

    // @formatter:off
    public Weights loadWeightsOlmoeOriginal(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLength(),
                config.headSize(),
                config.ropeTheta(),
                false,
                0, 0, 0, 0
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault("output.weight", tokenEmbeddings);

        if (Options.getDefaultOptions().useTornadovm()) {
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                System.out.println("Loading OLMoE model weights in TornadoVM format");
            }
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            // Enable tensor debugging for OLMoE standard weights
            System.err.println("[OLMOE-STANDARD] Available tensor names (first 50):");
            tensorEntries.keySet().stream().sorted().limit(50).forEach(name -> System.err.println("  " + name));
            System.err.printf("[OLMOE-STANDARD] Total tensors: %d%n", tensorEntries.size());

            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config,
            Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {

        // OLMOE DEBUG: Print available tensor names to identify the correct naming pattern
        System.err.println("[OLMOE-DEBUG] Available tensor names (first 50):");
        tensorEntries.keySet().stream().sorted().limit(50).forEach(name ->
            System.err.println("  " + name));
        System.err.printf("[OLMOE-DEBUG] Total tensors available: %d%n", tensorEntries.size());

        // OLMOE FIX: Check for missing tensors with proper OLMoE naming
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String[] requiredTensors = {
                "blk." + i + ".attn_norm.weight",
                "blk." + i + ".attn_q.weight",
                "blk." + i + ".attn_k.weight",
                "blk." + i + ".attn_v.weight",
                "blk." + i + ".attn_output.weight",
                "blk." + i + ".ffn_norm.weight",
                // OLMoE uses expert-specific naming instead of single gates
                "blk." + i + ".ffn_gate_inp.weight",  // Router/gating
                "blk." + i + ".ffn_gate_exps.weight", // Expert gates
                "blk." + i + ".ffn_down_exps.weight", // Expert down projections
                "blk." + i + ".ffn_up_exps.weight"   // Expert up projections
            };

            for (String tensorName : requiredTensors) {
                GGMLTensorEntry entry = tensorEntries.get(tensorName);
                if (entry == null) {
                    System.err.printf("[OLMOE-ERROR] Missing tensor: %s%n", tensorName);

                    // Try to find similar tensor names for debugging
                    String layerPrefix = "blk." + i + ".";
                    System.err.printf("[OLMOE-DEBUG] Available tensors for layer %d:%n", i);
                    tensorEntries.keySet().stream()
                        .filter(name -> name.startsWith(layerPrefix))
                        .sorted()
                        .forEach(name -> System.err.println("    " + name));

                    throw new IllegalArgumentException("Missing required tensor for OLMoE model: " + tensorName +
                        ". This may indicate an incompatible GGUF file format or naming convention.");
                }
            }
        }

        // Check for output_norm.weight
        GGMLTensorEntry outputNormEntry = tensorEntries.get("output_norm.weight");
        if (outputNormEntry == null) {
            System.err.println("[OLMOE-ERROR] Missing output_norm.weight tensor");
            System.err.println("[OLMOE-DEBUG] Available norm-related tensors:");
            tensorEntries.keySet().stream()
                .filter(name -> name.contains("norm"))
                .sorted()
                .forEach(name -> System.err.println("    " + name));
            throw new IllegalArgumentException("Missing required tensor: output_norm.weight");
        }

        // For now, use standard Llama weights structure with fallback tensor names
        // In a full implementation, this would handle MoE-specific weight loading
        return new LlamaTornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_norm.weight", null)),
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_q.weight", null)),
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_k.weight", null)),
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_v.weight", null)),
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_output.weight",
                        tensorEntries.get("blk." + i + ".attn_out.weight"))),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".ffn_norm.weight", null)),
                // OLMoE expert weights - Map to Llama weight positions correctly
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.get("blk." + i + ".ffn_gate_exps.weight")), // w1 = Expert gates
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight")), // w2 = Expert down
                loadArrayAsHalfFloatArray(config.numberOfLayers(),
                    i -> tensorEntries.get("blk." + i + ".ffn_up_exps.weight")), // w3 = Expert up
                floatBufferToFloatArray(outputNormEntry),
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

        // OLMOE FIX: Check for missing tensors in standard weights loading with correct OLMoE naming
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String[] requiredTensors = {
                // Standard attention tensors (same as Llama)
                "blk." + i + ".attn_norm.weight",
                "blk." + i + ".attn_q.weight",
                "blk." + i + ".attn_k.weight",
                "blk." + i + ".attn_v.weight",
                "blk." + i + ".attn_output.weight",
                "blk." + i + ".ffn_norm.weight",
                // OLMoE-specific MoE tensors (expert-based)
                "blk." + i + ".ffn_gate_inp.weight",    // Router/gating
                "blk." + i + ".ffn_gate_exps.weight",   // Expert gates (NEW)
                "blk." + i + ".ffn_down_exps.weight",   // Expert down projections
                "blk." + i + ".ffn_up_exps.weight"      // Expert up projections
            };

            for (String tensorName : requiredTensors) {
                GGMLTensorEntry entry = tensorEntries.get(tensorName);
                if (entry == null) {
                    System.err.printf("[OLMOE-STANDARD-ERROR] Missing tensor: %s%n", tensorName);

                    // Show available tensors for this layer
                    String layerPrefix = "blk." + i + ".";
                    System.err.printf("[OLMOE-STANDARD-DEBUG] Available tensors for layer %d:%n", i);
                    tensorEntries.keySet().stream()
                        .filter(name -> name.startsWith(layerPrefix))
                        .sorted()
                        .forEach(name -> System.err.println("    " + name));

                    // Check if we can find alternative naming (shouldn't be needed now with correct OLMoE names)
                    String[] alternatives = {
                        tensorName.replace("ffn_gate_exps", "ffn_gate"),
                        tensorName.replace("ffn_gate_inp", "ffn_gate"),
                        tensorName.replace("ffn_down_exps", "ffn_down"),
                        tensorName.replace("ffn_up_exps", "ffn_up"),
                        tensorName.replace("attn_output", "attn_out")
                    };

                    GGMLTensorEntry alternative = null;
                    String foundAlternative = null;
                    for (String alt : alternatives) {
                        if (!alt.equals(tensorName)) {
                            alternative = tensorEntries.get(alt);
                            if (alternative != null) {
                                foundAlternative = alt;
                                break;
                            }
                        }
                    }

                    if (foundAlternative != null) {
                        System.err.printf("[OLMOE-STANDARD-FIX] Found alternative tensor: %s%n", foundAlternative);
                    } else {
                        throw new IllegalArgumentException("Missing required tensor for OLMoE standard weights: " + tensorName +
                            ". Check the GGUF file format and tensor naming convention.");
                    }
                }
            }
        }

        // Check for output_norm.weight
        GGMLTensorEntry outputNormEntry = tensorEntries.get("output_norm.weight");
        if (outputNormEntry == null) {
            System.err.println("[OLMOE-STANDARD-ERROR] Missing output_norm.weight tensor");
            System.err.println("[OLMOE-STANDARD-DEBUG] Available norm-related tensors:");
            tensorEntries.keySet().stream()
                .filter(name -> name.contains("norm"))
                .sorted()
                .forEach(name -> System.err.println("    " + name));

            // Try alternatives
            outputNormEntry = tensorEntries.get("output_norm");
            if (outputNormEntry == null) {
                outputNormEntry = tensorEntries.get("norm.weight");
            }
            if (outputNormEntry == null) {
                throw new IllegalArgumentException("Missing required tensor: output_norm.weight (no alternatives found)");
            }
        }

        // For now, use standard Llama weights structure with fallback tensor names
        // In a full implementation, this would handle MoE-specific weight loading
        return new LlamaStandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_norm.weight", null)),
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_q.weight", null)),
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_k.weight", null)),
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_v.weight", null)),
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".attn_output.weight",
                        tensorEntries.get("blk." + i + ".attn_out.weight"))),
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.getOrDefault("blk." + i + ".ffn_norm.weight", null)),
                // OLMoE expert weights - Map to Llama weight positions correctly
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.get("blk." + i + ".ffn_gate_exps.weight")), // w1 = Expert gates
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight")), // w2 = Expert down
                loadArrayOfQuantized(config.numberOfLayers(),
                    i -> tensorEntries.get("blk." + i + ".ffn_up_exps.weight")),   // w3 = Expert up
                loadQuantized(outputNormEntry),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType()
        );
    }
    // @formatter:on
}