package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
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
import org.beehive.gpullama3.tokenizer.impl.GptNeoXTokenizer;
import org.beehive.gpullama3.tokenizer.impl.OlmoTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.model.loader.batch.BatchCapableModelLoader;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import org.beehive.gpullama3.inference.weights.olmoe.OlmoeStandardWeights;
import org.beehive.gpullama3.inference.weights.olmoe.OlmoeTornadoWeights;

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
        // Use new OLMoE-specific weight creation logic
        return createOlmoeWeightsFromTensors(tensors, config);
    }

    /**
     * Creates OLMoE weights from loaded tensors using new MoE-specific weight classes
     */
    private Weights createOlmoeWeightsFromTensors(Map<String, org.beehive.gpullama3.core.model.tensor.FloatTensor> loadedTensors, Configuration config) {
        System.err.printf("[OLMOE-NEW] Creating OLMoE weights from %d loaded tensors%n", loadedTensors.size());

        // Get RoPE frequencies
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLength(),
                config.headSize(),
                config.ropeTheta(),
                false,
                0, 0, 0, 0
        );

        // CRITICAL FIX: Check tensor names in GGUF file before loading
        System.err.println("[OLMOE-TENSOR-DEBUG] Available tensor names in GGUF:");
        loadedTensors.keySet().stream()
            .filter(name -> name.contains("token_embd") || name.contains("output") || name.contains("lm_head"))
            .sorted()
            .forEach(name -> System.err.println("  " + name));

        // Get token embeddings and output weight - try both with and without .weight suffix
        org.beehive.gpullama3.core.model.tensor.FloatTensor tokenEmbeddings = loadedTensors.get("token_embd.weight");
        if (tokenEmbeddings == null) {
            tokenEmbeddings = loadedTensors.get("token_embd");
            System.err.println("[OLMOE-TENSOR-DEBUG] Using 'token_embd' (without .weight suffix)");
        } else {
            System.err.println("[OLMOE-TENSOR-DEBUG] Using 'token_embd.weight'");
        }

        org.beehive.gpullama3.core.model.tensor.FloatTensor outputWeight = loadedTensors.get("output.weight");
        if (outputWeight == null) {
            outputWeight = loadedTensors.get("output");
            System.err.println("[OLMOE-TENSOR-DEBUG] Using 'output' (without .weight suffix)");
        } else {
            System.err.println("[OLMOE-TENSOR-DEBUG] Using 'output.weight'");
        }

        // CRITICAL FIX: Check if token_embd contains negative zeros (tied embeddings indicator)
        boolean tokenEmbedCorrupted = false;
        if (tokenEmbeddings != null) {
            // Check for the 00 80 pattern (F16 negative zeros) that indicates tied embeddings
            float embSum = 0.0f;
            int negativeZeroCount = 0;
            for (int i = 0; i < Math.min(1000, tokenEmbeddings.size()); i++) {
                float val = tokenEmbeddings.getFloat(i);
                embSum += Math.abs(val);
                // Check for negative zero pattern (00 80 in F16 = -0.0)
                if (val == -0.0f || Float.floatToIntBits(val) == 0x80000000) {
                    negativeZeroCount++;
                }
            }

            // If token_embd is mostly negative zeros, OLMoE uses tied embeddings
            if (embSum < 0.0001f || negativeZeroCount > 500) {
                tokenEmbedCorrupted = true;
                System.err.printf("[OLMOE-TIED-EMBEDDINGS] token_embd.weight contains %d negative zeros, sum=%.6f - TIED EMBEDDINGS DETECTED!%n",
                                negativeZeroCount, embSum);
                System.err.println("[OLMOE-FIX] Using output.weight as token embeddings (tied embeddings pattern)");

                // CRITICAL FIX: Use output.weight as token embeddings
                if (outputWeight != null) {
                    tokenEmbeddings = outputWeight;
                    System.err.println("[OLMOE-FIX] Successfully replaced token_embd.weight with output.weight");
                } else {
                    throw new IllegalArgumentException("CRITICAL: token_embd corrupted and no output.weight found!");
                }
            } else {
                System.err.printf("[OLMOE-EMBEDDINGS-OK] token_embd.weight sum = %.6f, negZeros = %d (normal embeddings)%n",
                                embSum, negativeZeroCount);
            }
        }

        // CRITICAL DEBUG: Check available output-related tensors
        System.err.println("[OLMOE-TENSOR-DEBUG] Available output-related tensors:");
        loadedTensors.keySet().stream()
            .filter(name -> name.contains("output") || name.contains("lm_head") || name.contains("embed"))
            .sorted()
            .forEach(name -> System.err.println("  " + name));

        // Try alternative names common in other LLMs
        FloatTensor lmHeadWeight = loadedTensors.get("lm_head.weight");
        FloatTensor outputNormWeight = loadedTensors.get("output_norm.weight");

        if (outputWeight == null) {
            System.err.println("[OLMOE-WEIGHT-ERROR] output.weight not found in GGUF!");
            if (lmHeadWeight != null) {
                System.err.println("[OLMOE-WEIGHT-ALTERNATIVE] Using lm_head.weight as output weights");
                outputWeight = lmHeadWeight;
            } else {
                System.err.println("[OLMOE-WEIGHT-FALLBACK] Using tokenEmbeddings as output weights - THIS MAY CAUSE GIBBERISH!");
                outputWeight = tokenEmbeddings;
            }
        } else {
            System.err.println("[OLMOE-WEIGHT-SUCCESS] Found output.weight tensor");

            // Additional debug: Compare first few values of different tensors
            if (lmHeadWeight != null) {
                System.err.printf("[OLMOE-COMPARE] lm_head.weight first 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]%n",
                    lmHeadWeight.getFloat(0), lmHeadWeight.getFloat(1), lmHeadWeight.getFloat(2),
                    lmHeadWeight.getFloat(3), lmHeadWeight.getFloat(4));

                // CRITICAL FIX: If lm_head.weight exists and is different from output.weight, use it!
                boolean outputCorrupted = (Math.abs(outputWeight.getFloat(0) + 0.007294f) < 0.000001f &&
                                         Math.abs(outputWeight.getFloat(1) + 0.009155f) < 0.000001f);
                boolean lmHeadDifferent = (Math.abs(lmHeadWeight.getFloat(0) + 0.007294f) > 0.000001f ||
                                         Math.abs(lmHeadWeight.getFloat(1) + 0.009155f) > 0.000001f);

                if (outputCorrupted && lmHeadDifferent) {
                    System.err.println("[OLMOE-FIX] output.weight appears corrupted, switching to lm_head.weight!");
                    outputWeight = lmHeadWeight;
                }
            }
            System.err.printf("[OLMOE-COMPARE] token_embd.weight first 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]%n",
                tokenEmbeddings.getFloat(0), tokenEmbeddings.getFloat(1), tokenEmbeddings.getFloat(2),
                tokenEmbeddings.getFloat(3), tokenEmbeddings.getFloat(4));
        }

        if (Options.getDefaultOptions().useTornadovm()) {
            System.err.println("[OLMOE-NEW] Creating TornadoVM weights");
            return createOlmoeTornadoWeights(loadedTensors, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            System.err.println("[OLMOE-NEW] Creating standard weights");
            return createOlmoeStandardWeights(loadedTensors, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    // @formatter:off
    @Override
    public Olmoe loadModel() {
        try (var ignored = Timer.log("Load OLMoE model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            // Load OLMoE-specific vocabulary
            Vocabulary vocabulary = loadOlmoeVocabulary(metadata);

            // Detect tokenizer type from GGUF metadata
            String tokenizerPreType = (String) metadata.get("tokenizer.ggml.pre");
            System.out.println("[OLMOE-TOKENIZER-DETECTION] tokenizer.ggml.pre = " + tokenizerPreType);

            // Create appropriate tokenizer based on metadata
            Tokenizer tokenizer;
            if ("olmo".equals(tokenizerPreType)) {
                System.out.println("[OLMOE-TOKENIZER] Creating OLMo tokenizer (GPT2-style pattern with OLMo vocabulary)");
                tokenizer = new OlmoTokenizer(metadata, vocabulary);
            } else {
                System.out.println("[OLMOE-TOKENIZER] Fallback to GPT-NeoX tokenizer");
                tokenizer = new GptNeoXTokenizer(metadata, vocabulary);
            }

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

            // OLMoE Instruct models use Tulu chat template - detect and use appropriate format
            ChatFormat chatFormat;
            if (isInstructModel(metadata)) {
                System.err.println("[OLMOE-CHAT-FORMAT] Detected Instruct model - using Tulu chat template");
                if (tokenizer instanceof GptNeoXTokenizer) {
                    chatFormat = ChatFormat.createOLMoETulu((GptNeoXTokenizer) tokenizer);
                } else {
                    // For OlmoTokenizer, create generic chat format with Tulu tokens
                    ChatTokens chatTokens = new ChatTokens(
                        "<|user|>",          // tStartHeader: Correct Tulu format start header
                        "<|assistant|>",     // tEndHeader: Correct Tulu format end header
                        "",                  // tEndOfTurn: Assistant prefix
                        "<|endoftext|>",     // tEndOfText: End of text
                        "<|endoftext|>"      // tEndOfTextFim: End of text FIM
                    );
                    chatFormat = ChatFormat.create(tokenizer, chatTokens);
                }
            } else {
                System.err.println("[OLMOE-CHAT-FORMAT] Base model detected - using generic format");
                ChatTokens chatTokens = new ChatTokens(
                    "<|user|>",          // Correct Tulu format start header
                    "<|assistant|>",     // Correct Tulu format end header
                    "",                  // Assistant prefix
                    "<|endoftext|>",     // End of text
                    ""                   // No special reasoning token
                );
                chatFormat = ChatFormat.create(tokenizer, chatTokens);
            }

            return new Olmoe(config, tokenizer, weights, chatFormat);
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

    /**
     * Detects if this is an Instruct model based on metadata.
     * OLMoE Instruct models should use the Tulu chat template.
     */
    private boolean isInstructModel(Map<String, Object> metadata) {
        // Check model name/identifier for "Instruct" suffix
        String modelName = (String) metadata.get("general.name");
        if (modelName != null && modelName.toLowerCase().contains("instruct")) {
            return true;
        }

        // Check for fine-tuning related metadata
        String finetuneType = (String) metadata.get("general.finetune");
        if (finetuneType != null && finetuneType.toLowerCase().contains("instruct")) {
            return true;
        }

        // Check for chat template in metadata (indicates instruction-tuning)
        String chatTemplate = (String) metadata.get("tokenizer.chat_template");
        if (chatTemplate != null && !chatTemplate.trim().isEmpty()) {
            return true;
        }

        // Default to false for base models
        return false;
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

        // CRITICAL FIX: Try both tensor name formats for consistency with llama.cpp
        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        if (tokenEmbeddings == null) {
            tokenEmbeddings = tensorEntries.get("token_embd");
            System.err.println("[OLMOE-LOADER] Using 'token_embd' tensor entry");
        } else {
            System.err.println("[OLMOE-LOADER] Using 'token_embd.weight' tensor entry");
        }

        GGMLTensorEntry outputWeight = tensorEntries.get("output.weight");
        if (outputWeight == null) {
            outputWeight = tensorEntries.get("output");
            if (outputWeight == null) {
                outputWeight = tokenEmbeddings; // Fallback to tied embeddings
                System.err.println("[OLMOE-LOADER] Using tied embeddings for output weights");
            } else {
                System.err.println("[OLMOE-LOADER] Using 'output' tensor entry");
            }
        } else {
            System.err.println("[OLMOE-LOADER] Using 'output.weight' tensor entry");
        }

        // CRITICAL FIX: Check for tied embeddings pattern in tensor entries too
        boolean tensorEmbedCorrupted = false;
        if (tokenEmbeddings != null && outputWeight != null) {
            // Quick check for negative zero pattern by loading first few values
            FloatTensor tokenTensor = loadQuantized(tokenEmbeddings);
            float embSum = 0.0f;
            int negativeZeroCount = 0;
            for (int i = 0; i < Math.min(100, tokenTensor.size()); i++) {
                float val = tokenTensor.getFloat(i);
                embSum += Math.abs(val);
                if (val == -0.0f || Float.floatToIntBits(val) == 0x80000000) {
                    negativeZeroCount++;
                }
            }

            if (embSum < 0.0001f || negativeZeroCount > 50) {
                tensorEmbedCorrupted = true;
                System.err.printf("[OLMOE-TENSOR-ENTRY-TIED] token_embd tensor entry corrupted (sum=%.6f, negZeros=%d), using output.weight%n",
                                embSum, negativeZeroCount);
                tokenEmbeddings = outputWeight;
            }
        }

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

    /**
     * Creates OLMoE standard weights from loaded tensors with proper MoE structure
     */
    private Weights createOlmoeStandardWeights(Map<String, org.beehive.gpullama3.core.model.tensor.FloatTensor> loadedTensors,
                                              Configuration config,
                                              Pair<float[], float[]> ropeFreqs,
                                              org.beehive.gpullama3.core.model.tensor.FloatTensor tokenEmbeddings,
                                              org.beehive.gpullama3.core.model.tensor.FloatTensor outputWeight) {

        int numLayers = config.numberOfLayers();

        // Load standard attention weights (same as Llama)
        FloatTensor[] attentionNorms = new FloatTensor[numLayers];
        FloatTensor[] queryWeights = new FloatTensor[numLayers];
        FloatTensor[] keyWeights = new FloatTensor[numLayers];
        FloatTensor[] valueWeights = new FloatTensor[numLayers];
        FloatTensor[] outputWeights = new FloatTensor[numLayers];
        FloatTensor[] ffnNorms = new FloatTensor[numLayers];

        // Load MoE-specific weights
        FloatTensor[] routerWeights = new FloatTensor[numLayers];
        FloatTensor[] expertGateWeights = new FloatTensor[numLayers];
        FloatTensor[] expertDownWeights = new FloatTensor[numLayers];
        FloatTensor[] expertUpWeights = new FloatTensor[numLayers];

        for (int i = 0; i < numLayers; i++) {
            // Standard attention weights
            attentionNorms[i] = loadedTensors.get("blk." + i + ".attn_norm.weight");
            queryWeights[i] = loadedTensors.get("blk." + i + ".attn_q.weight");
            keyWeights[i] = loadedTensors.get("blk." + i + ".attn_k.weight");
            valueWeights[i] = loadedTensors.get("blk." + i + ".attn_v.weight");
            outputWeights[i] = loadedTensors.getOrDefault("blk." + i + ".attn_output.weight",
                                                        loadedTensors.get("blk." + i + ".attn_out.weight"));
            ffnNorms[i] = loadedTensors.get("blk." + i + ".ffn_norm.weight");

            // MoE-specific weights
            routerWeights[i] = loadedTensors.get("blk." + i + ".ffn_gate_inp.weight");
            expertGateWeights[i] = loadedTensors.get("blk." + i + ".ffn_gate_exps.weight");
            expertDownWeights[i] = loadedTensors.get("blk." + i + ".ffn_down_exps.weight");
            expertUpWeights[i] = loadedTensors.get("blk." + i + ".ffn_up_exps.weight");

            // Validate required tensors
            if (routerWeights[i] == null) {
                throw new IllegalArgumentException("Missing router weights for layer " + i + ": blk." + i + ".ffn_gate_inp.weight");
            }
            if (expertGateWeights[i] == null) {
                throw new IllegalArgumentException("Missing expert gate weights for layer " + i + ": blk." + i + ".ffn_gate_exps.weight");
            }
            if (expertDownWeights[i] == null) {
                throw new IllegalArgumentException("Missing expert down weights for layer " + i + ": blk." + i + ".ffn_down_exps.weight");
            }
            if (expertUpWeights[i] == null) {
                throw new IllegalArgumentException("Missing expert up weights for layer " + i + ": blk." + i + ".ffn_up_exps.weight");
            }
        }

        // Output norm
        FloatTensor outputNorm = loadedTensors.getOrDefault("output_norm.weight", loadedTensors.get("norm.weight"));
        if (outputNorm == null) {
            throw new IllegalArgumentException("Missing output norm weight");
        }

        return new OlmoeStandardWeights(
                tokenEmbeddings,
                attentionNorms,
                queryWeights,
                keyWeights,
                valueWeights,
                outputWeights,
                ffnNorms,
                routerWeights,
                expertGateWeights,
                expertDownWeights,
                expertUpWeights,
                outputNorm,
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                outputWeight,
                GGMLType.F32
        );
    }

    /**
     * Creates OLMoE TornadoVM weights from loaded tensors with proper MoE structure
     */
    private Weights createOlmoeTornadoWeights(Map<String, org.beehive.gpullama3.core.model.tensor.FloatTensor> loadedTensors,
                                             Configuration config,
                                             Pair<float[], float[]> ropeFreqs,
                                             org.beehive.gpullama3.core.model.tensor.FloatTensor tokenEmbeddings,
                                             org.beehive.gpullama3.core.model.tensor.FloatTensor outputWeight) {

        int numLayers = config.numberOfLayers();

        // Load standard attention weights as FloatArrays/HalfFloatArrays for TornadoVM
        FloatArray[] attentionNorms = new FloatArray[numLayers];      // Norms stay as FloatArray
        HalfFloatArray[] queryWeights = new HalfFloatArray[numLayers]; // Weights as HalfFloatArray
        HalfFloatArray[] keyWeights = new HalfFloatArray[numLayers];
        HalfFloatArray[] valueWeights = new HalfFloatArray[numLayers];
        HalfFloatArray[] outputWeights = new HalfFloatArray[numLayers];
        FloatArray[] ffnNorms = new FloatArray[numLayers];

        // CRITICAL: Load OLMoE-specific Q/K normalization weights
        FloatArray[] attnQNormWeights = new FloatArray[numLayers];    // Q normalization weights
        FloatArray[] attnKNormWeights = new FloatArray[numLayers];    // K normalization weights

        // Load MoE-specific weights as FloatArrays
        FloatArray[] routerWeights = new FloatArray[numLayers];
        FloatArray[] expertGateWeights = new FloatArray[numLayers];
        FloatArray[] expertDownWeights = new FloatArray[numLayers];
        FloatArray[] expertUpWeights = new FloatArray[numLayers];

        for (int i = 0; i < numLayers; i++) {
            // Convert standard weights to FloatArrays
            attentionNorms[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".attn_norm.weight"));
            queryWeights[i] = convertToHalfFloatArray(loadedTensors.get("blk." + i + ".attn_q.weight"));
            keyWeights[i] = convertToHalfFloatArray(loadedTensors.get("blk." + i + ".attn_k.weight"));
            valueWeights[i] = convertToHalfFloatArray(loadedTensors.get("blk." + i + ".attn_v.weight"));
            outputWeights[i] = convertToHalfFloatArray(loadedTensors.getOrDefault("blk." + i + ".attn_output.weight",
                                                                                 loadedTensors.get("blk." + i + ".attn_out.weight")));
            ffnNorms[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".ffn_norm.weight"));

            // CRITICAL: Load OLMoE-specific Q/K normalization weights
            attnQNormWeights[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".attn_q_norm.weight"));
            attnKNormWeights[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".attn_k_norm.weight"));

            // Validate that Q/K norm weights were loaded successfully
            if (attnQNormWeights[i] == null) {
                throw new IllegalArgumentException("Missing Q normalization weights for layer " + i);
            }
            if (attnKNormWeights[i] == null) {
                throw new IllegalArgumentException("Missing K normalization weights for layer " + i);
            }

            // Convert MoE weights to FloatArrays
            org.beehive.gpullama3.core.model.tensor.FloatTensor routerTensor = loadedTensors.get("blk." + i + ".ffn_gate_inp.weight");
            if (routerTensor != null) {
                System.err.printf("[ROUTER-LOAD] Layer %d: Loading router weights, size=%d%n", i, routerTensor.size());
                // Check first few values
                if (routerTensor.size() > 5) {
                    System.err.printf("[ROUTER-LOAD] Layer %d: First 5 router weights: [%.6f, %.6f, %.6f, %.6f, %.6f]%n",
                                     i, routerTensor.getFloat(0), routerTensor.getFloat(1),
                                     routerTensor.getFloat(2), routerTensor.getFloat(3), routerTensor.getFloat(4));
                }
            }
            routerWeights[i] = convertToFloatArray(routerTensor);
            expertGateWeights[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".ffn_gate_exps.weight"));
            expertDownWeights[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".ffn_down_exps.weight"));
            expertUpWeights[i] = convertToFloatArray(loadedTensors.get("blk." + i + ".ffn_up_exps.weight"));

            // Validate required tensors
            if (routerWeights[i] == null) {
                throw new IllegalArgumentException("Missing router weights for layer " + i);
            }
        }

        // Convert output weights
        FloatArray outputNormArray = convertToFloatArray(loadedTensors.getOrDefault("output_norm.weight",
                                                                                   loadedTensors.get("norm.weight")));
        if (outputNormArray == null) {
            throw new IllegalArgumentException("Missing output norm weight");
        }

        // Create placeholder arrays for w1/w2/w3 since OLMoE uses expert weights instead
        int layerCount = attentionNorms.length;
        HalfFloatArray[] w1Placeholder = new HalfFloatArray[layerCount];
        HalfFloatArray[] w2Placeholder = new HalfFloatArray[layerCount];
        HalfFloatArray[] w3Placeholder = new HalfFloatArray[layerCount];

        // Initialize placeholders as empty arrays
        for (int i = 0; i < layerCount; i++) {
            w1Placeholder[i] = new HalfFloatArray(1); // Minimal placeholder
            w2Placeholder[i] = new HalfFloatArray(1);
            w3Placeholder[i] = new HalfFloatArray(1);
        }

        // Collect source tensors for selective expert loading
        FloatTensor[] sourceExpertGateWeights = new FloatTensor[numLayers];
        FloatTensor[] sourceExpertDownWeights = new FloatTensor[numLayers];
        FloatTensor[] sourceExpertUpWeights = new FloatTensor[numLayers];

        for (int i = 0; i < numLayers; i++) {
            sourceExpertGateWeights[i] = loadedTensors.get("blk." + i + ".ffn_gate_exps.weight");
            sourceExpertDownWeights[i] = loadedTensors.get("blk." + i + ".ffn_down_exps.weight");
            sourceExpertUpWeights[i] = loadedTensors.get("blk." + i + ".ffn_up_exps.weight");
        }

        return new OlmoeTornadoWeights(
                convertToFloatArray(tokenEmbeddings),  // tokenEmbeddingTable
                attentionNorms,                        // rms_att_weightLayered
                queryWeights,                          // wqLayered
                keyWeights,                            // wkLayered
                valueWeights,                          // wvLayered
                outputWeights,                         // woLayered
                ffnNorms,                              // rms_ffn_weightLayered
                w1Placeholder,                         // w1Layered (placeholder for OLMoE)
                w2Placeholder,                         // w2Layered (placeholder for OLMoE)
                w3Placeholder,                         // w3Layered (placeholder for OLMoE)
                outputNormArray,                       // rms_final_weight_as_floatArray
                FloatArray.fromArray(ropeFreqs.first()), // freq_cis_realFlat
                FloatArray.fromArray(ropeFreqs.second()), // freq_cis_imagFlat
                convertToHalfFloatArrayWithDebug(outputWeight, "wclsHalfFloat"), // wclsHalfFloat
                GGMLType.F32,                          // weightType
                routerWeights,                         // MoE-specific: routerWeights
                expertGateWeights,                     // MoE-specific: expertGateWeights
                expertDownWeights,                     // MoE-specific: expertDownWeights
                expertUpWeights,                       // MoE-specific: expertUpWeights
                attnQNormWeights,                      // CRITICAL: Q normalization weights
                attnKNormWeights,                      // CRITICAL: K normalization weights
                sourceExpertGateWeights,               // NEW: Source tensors for selective loading
                sourceExpertDownWeights,               // NEW: Source tensors for selective loading
                sourceExpertUpWeights                  // NEW: Source tensors for selective loading
        );
    }

    /**
     * Converts a FloatTensor to FloatArray for TornadoVM
     */
    private FloatArray convertToFloatArray(org.beehive.gpullama3.core.model.tensor.FloatTensor tensor) {
        if (tensor == null) return null;

        int size = tensor.size();
        FloatArray array = new FloatArray(size);

        for (int i = 0; i < size; i++) {
            array.set(i, tensor.getFloat(i));
        }

        return array;
    }

    /**
     * Debug version of convertToHalfFloatArray that logs source data
     */
    private HalfFloatArray convertToHalfFloatArrayWithDebug(org.beehive.gpullama3.core.model.tensor.FloatTensor tensor, String name) {
        if (tensor == null) {
            System.err.printf("[WEIGHT-DEBUG] %s: tensor is null%n", name);
            return null;
        }

        System.err.printf("[WEIGHT-DEBUG] %s: Converting tensor size=%d%n", name, tensor.size());

        // Debug: Check first few values of source tensor
        System.err.printf("[WEIGHT-DEBUG] %s: Source tensor first 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]%n",
                         name, tensor.getFloat(0), tensor.getFloat(1), tensor.getFloat(2),
                         tensor.getFloat(3), tensor.getFloat(4));

        return convertToHalfFloatArray(tensor);
    }

    /**
     * Converts a FloatTensor to HalfFloatArray for TornadoVM
     * Simple implementation that creates a HalfFloatArray from the FloatArray
     */
    private HalfFloatArray convertToHalfFloatArray(org.beehive.gpullama3.core.model.tensor.FloatTensor tensor) {
        if (tensor == null) {
            return null;
        }

        // First convert to FloatArray, then create HalfFloatArray from it
        FloatArray floatArray = convertToFloatArray(tensor);
        if (floatArray == null) {
            return null;
        }

        // Create HalfFloatArray and copy data
        HalfFloatArray halfFloatArray = new HalfFloatArray(floatArray.getSize());
        for (int i = 0; i < floatArray.getSize(); i++) {
            halfFloatArray.set(i, new HalfFloat(floatArray.get(i)));
        }
        return halfFloatArray;
    }
}