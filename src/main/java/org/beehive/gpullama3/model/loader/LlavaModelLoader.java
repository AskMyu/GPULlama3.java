package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.llava.Llava;
import org.beehive.gpullama3.model.llava.LlavaConfiguration;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.LlamaTokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.inference.weights.Weights;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * Model loader for LLaVA (Large Language and Vision Assistant) models.
 * Handles both the language model (Llama-3-8B) and the vision projector (mmproj) components.
 * 
 * LLaVA Architecture:
 * - Vision Encoder: CLIP-ViT-Large-patch14-336 
 * - Language Model: Llama-3-8B-Instruct
 * - Vision-Language Connector: MLP projector (mmproj file with 377 vision tensors)
 */
public class LlavaModelLoader extends ModelLoader {
    
    public LlavaModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    public Model loadModel() {
        try {
            Map<String, Object> metadata = gguf.getMetadata();
            
            // Load tensor entries from GGUF
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(
                fileChannel,
                gguf.getTensorDataOffset(),
                gguf.getTensorInfos()
            );

            // Debug: Show available tensor names to understand model structure
            System.err.println("DEBUG: Available tensor names in model:");
            tensorEntries.keySet().stream().sorted().limit(20).forEach(name ->
                System.err.println("  " + name)
            );
            System.err.println("  ... (showing first 20 of " + tensorEntries.size() + " total tensors)");
            
            // Extract LLaVA-specific configuration from GGUF metadata
            LlavaConfiguration config = extractConfiguration(metadata);
            
            System.out.println("Loading LLaVA model:");
            System.out.println("  Language Model: " + config.getLanguageModelPath());
            System.out.println("  Vision Projector: " + config.getVisionProjectorPath());
            System.out.println("  Vision Tokens: " + config.getVisionTokenCount());
            
            // Create tokenizer - LLaVA uses Llama tokenizer
            Tokenizer tokenizer = createLlamaTokenizer(tensorEntries, config);
            
            // Load vision projector weights if available
            Map<String, GGMLTensorEntry> visionTensorEntries = loadVisionProjector(config.getVisionProjectorPath(), tensorEntries);
            
            // Check if this is a Phi-2 based model and adapt tensor entries
            Map<String, GGMLTensorEntry> adaptedTensorEntries = adaptTensorEntriesForPhi2(tensorEntries, metadata);

            // Create and return LLaVA model instance
            return new Llava(config, tokenizer, adaptedTensorEntries, visionTensorEntries,
                           loadWeights ? loadWeights(adaptedTensorEntries, config) : null);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load LLaVA model", e);
        }
    }

    /**
     * Adapt tensor entries for Phi-2 based models to work with Llama weight loading.
     * Maps Phi-2 tensor names to expected Llama tensor names.
     */
    private Map<String, GGMLTensorEntry> adaptTensorEntriesForPhi2(Map<String, GGMLTensorEntry> originalEntries, Map<String, Object> metadata) {
        String architecture = (String) metadata.get("general.architecture");

        if (!"phi2".equals(architecture)) {
            // Not a Phi-2 model, return original entries
            return originalEntries;
        }

        System.err.println("DEBUG: Adapting Phi-2 tensor names to Llama format");
        Map<String, GGMLTensorEntry> adaptedEntries = new java.util.HashMap<>(originalEntries);

        // Get number of layers from metadata
        Integer numberOfLayers = (Integer) metadata.get("phi2.block_count");
        if (numberOfLayers == null) {
            System.err.println("WARNING: Could not get number of layers for Phi-2 model");
            return originalEntries;
        }

        // Map Phi-2 tensor names to Llama equivalents
        for (int i = 0; i < numberOfLayers; i++) {
            // Map ffn_norm: Use attn_norm as ffn_norm (Phi-2 doesn't have separate FFN norm)
            String attnNormKey = "blk." + i + ".attn_norm.weight";
            String ffnNormKey = "blk." + i + ".ffn_norm.weight";
            if (originalEntries.containsKey(attnNormKey) && !originalEntries.containsKey(ffnNormKey)) {
                adaptedEntries.put(ffnNormKey, originalEntries.get(attnNormKey));
                System.err.println("DEBUG: Mapped " + attnNormKey + " -> " + ffnNormKey);
            }

            // Map ffn_gate: Use ffn_up as ffn_gate (Phi-2 uses different naming)
            String ffnUpKey = "blk." + i + ".ffn_up.weight";
            String ffnGateKey = "blk." + i + ".ffn_gate.weight";
            if (originalEntries.containsKey(ffnUpKey) && !originalEntries.containsKey(ffnGateKey)) {
                adaptedEntries.put(ffnGateKey, originalEntries.get(ffnUpKey));
                System.err.println("DEBUG: Mapped " + ffnUpKey + " -> " + ffnGateKey);
            }
        }

        // Check if we need output_norm mapping
        String outputNormKey = "output_norm.weight";
        if (!originalEntries.containsKey(outputNormKey)) {
            // Try to find an equivalent output normalization tensor
            String[] candidates = {"norm.weight", "ln_f.weight", "final_layernorm.weight"};
            for (String candidate : candidates) {
                if (originalEntries.containsKey(candidate)) {
                    adaptedEntries.put(outputNormKey, originalEntries.get(candidate));
                    System.err.println("DEBUG: Mapped " + candidate + " -> " + outputNormKey);
                    break;
                }
            }
        }

        System.err.println("DEBUG: Tensor adaptation complete. Original: " + originalEntries.size() + ", Adapted: " + adaptedEntries.size());
        return adaptedEntries;
    }
    
    private Tokenizer createLlamaTokenizer(Map<String, GGMLTensorEntry> tensorEntries, LlavaConfiguration config) {
        // Load vocabulary from GGUF metadata (same as regular Llama models)
        Map<String, Object> metadata = gguf.getMetadata();
        Vocabulary vocabulary = Vocabulary.loadLlamaVocabulary(metadata);
        return new LlamaTokenizer(metadata, vocabulary);
    }

    /**
     * Extract LLaVA configuration from GGUF metadata.
     */
    private LlavaConfiguration extractConfiguration(Map<String, Object> metadata) {
        // Extract base Llama configuration 
        LlamaConfiguration baseConfig = extractLlamaConfiguration(metadata);
        
        // LLaVA-specific parameters - extract model path from metadata debug context
        String modelName = (String) metadata.getOrDefault("general.name", "llava-llama-3-8b");
        String filename = null;
        
        // Try to get the model path from various sources
        if (this.modelPath != null) {
            filename = this.modelPath;
        } else {
            // Try to extract from FileChannel path
            try {
                if (fileChannel instanceof java.nio.channels.FileChannel) {
                    java.lang.reflect.Method method = fileChannel.getClass().getDeclaredMethod("path");
                    method.setAccessible(true);
                    java.nio.file.Path path = (java.nio.file.Path) method.invoke(fileChannel);
                    if (path != null) {
                        filename = path.toString();
                        System.out.println("DEBUG: Extracted filename from FileChannel: " + filename);
                    }
                }
            } catch (Exception e) {
                System.err.println("DEBUG: Could not extract path from FileChannel: " + e.getMessage());
            }

            // Final fallback
            if (filename == null) {
                filename = "models/vlm/llava-1.5-7b-gguf/llava-v1.5-7b-Q4_K_M.gguf"; // Default to LLaVA-1.5
                System.out.println("DEBUG: Using default filename: " + filename);
            }
        }
        
        // For LLaVA-1.5, general.name is "LLaMA v2" which isn't helpful, use filename instead
        // For LLaVA-Phi, general.name is "Phi2" which isn't helpful, use filename instead
        if ("LLaMA v2".equals(modelName) || "llama".equals(modelName) || "Phi2".equals(modelName)) {
            modelName = filename;
            System.out.println("DEBUG: Using filename as model name: " + modelName);
        }
        
        boolean isInt4 = modelName.toLowerCase().contains("int4") || modelName.toLowerCase().contains("q4");
        
        // Vision configuration
        int visionTokenCount = 576; // CLIP-ViT-Large-patch14-336: 24x24 patches = 576 tokens
        String visionProjectorPath = inferVisionProjectorPath(modelName);
        
        return LlavaConfiguration.builder()
            .fromLlamaConfig(baseConfig)
            .visionTokenCount(visionTokenCount)
            .visionProjectorPath(visionProjectorPath)
            .languageModelPath(fileChannel.toString())
            .isQuantizedInt4(isInt4)
            .build();
    }

    /**
     * Extract base Llama configuration from metadata.
     */
    private LlamaConfiguration extractLlamaConfiguration(Map<String, Object> metadata) {
        // Debug: Print all available metadata keys to understand LLaVA structure
        System.out.println("DEBUG: Available metadata keys for LLaVA model:");
        metadata.keySet().stream().sorted().forEach(key ->
            System.out.println("  " + key + " = " + metadata.get(key))
        );

        // Detect architecture and extract parameters accordingly
        String architecture = (String) metadata.get("general.architecture");

        int dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads;

        if ("phi2".equals(architecture)) {
            // Extract Phi-2 configuration parameters
            dim = ((Number) metadata.get("phi2.embedding_length")).intValue();
            hiddenDim = ((Number) metadata.get("phi2.feed_forward_length")).intValue();
            numberOfLayers = ((Number) metadata.get("phi2.block_count")).intValue();
            numberOfHeads = ((Number) metadata.get("phi2.attention.head_count")).intValue();
            numberOfKeyValueHeads = ((Number) metadata.get("phi2.attention.head_count_kv")).intValue();
        } else {
            // Extract standard Llama-3 configuration parameters with fallbacks for LLaVA-1.5
            dim = ((Number) metadata.get("llama.embedding_length")).intValue();
            hiddenDim = ((Number) metadata.get("llama.feed_forward_length")).intValue();
            numberOfLayers = ((Number) metadata.get("llama.block_count")).intValue();
            numberOfHeads = ((Number) metadata.get("llama.attention.head_count")).intValue();
            numberOfKeyValueHeads = ((Number) metadata.get("llama.attention.head_count_kv")).intValue();
        }
        
        // Extract vocabulary size - try architecture-specific fields first
        Number vocabSizeNum = null;
        if ("phi2".equals(architecture)) {
            vocabSizeNum = (Number) metadata.get("phi2.vocab_size");
        } else {
            vocabSizeNum = (Number) metadata.get("llama.vocab_size");
        }

        // Fallback to tokenizer fields if architecture-specific not found
        if (vocabSizeNum == null) {
            vocabSizeNum = (Number) metadata.get("tokenizer.ggml.vocab_size");
        }
        if (vocabSizeNum == null) {
            vocabSizeNum = (Number) metadata.get("vocab_size");
        }
        if (vocabSizeNum == null) {
            // Try to get vocabulary size from tokenizer arrays
            Object tokens = metadata.get("tokenizer.ggml.tokens");
            if (tokens instanceof String[]) {
                vocabSizeNum = ((String[]) tokens).length;
                System.out.println("DEBUG: Got vocabulary size from tokenizer.ggml.tokens array: " + vocabSizeNum);
            }
        }
        if (vocabSizeNum == null) {
            System.err.println("WARNING: No vocabulary size found, using default based on architecture");
            vocabSizeNum = "phi2".equals(architecture) ? 50257 : 32000; // Phi-2 uses GPT-2 tokenizer (50257), Llama uses 32000
        }
        int vocabularySize = vocabSizeNum.intValue();

        // Extract context length based on architecture
        int contextLength;
        if ("phi2".equals(architecture)) {
            contextLength = ((Number) metadata.get("phi2.context_length")).intValue();
        } else {
            contextLength = ((Number) metadata.get("llama.context_length")).intValue();
        }

        // Extract RMS norm epsilon and RoPE theta based on architecture
        float rmsNormEps, ropeTheta;
        if ("phi2".equals(architecture)) {
            rmsNormEps = ((Number) metadata.getOrDefault("phi2.attention.layer_norm_epsilon", 1e-5f)).floatValue();
            ropeTheta = ((Number) metadata.getOrDefault("phi2.rope.freq_base", 10000.0f)).floatValue();
        } else {
            rmsNormEps = ((Number) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f)).floatValue();
            ropeTheta = ((Number) metadata.getOrDefault("llama.rope.freq_base", 500000.0f)).floatValue();
        }

        return new LlamaConfiguration(
            dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads,
            vocabularySize, contextLength, rmsNormEps, ropeTheta
        );
    }

    /**
     * Infer the vision projector path from the language model name.
     */
    private String inferVisionProjectorPath(String languageModelName) {
        // For LLaVA models, the mmproj file should be in the same directory structure
        String baseName = languageModelName.toLowerCase();
        String fullPath = languageModelName.toLowerCase();

        // Check if this is an integrated LLaVA model (single GGUF with vision components)
        if (fullPath.contains("llava-phi") || fullPath.contains("llava") || baseName.contains("llava") || baseName.contains("downloads")) {
            
            // First, try to find mmproj file in the same directory as the model
            Path modelPath = Paths.get(languageModelName);
            if (modelPath.getParent() != null) {
                Path parentDir = modelPath.getParent();
                // Look for any mmproj file in the same directory
                try {
                    Path mmprojInSameDir = Files.list(parentDir)
                        .filter(p -> p.getFileName().toString().toLowerCase().contains("mmproj"))
                        .findFirst()
                        .orElse(null);
                    
                    if (mmprojInSameDir != null && Files.exists(mmprojInSameDir)) {
                        System.err.println("Found mmproj file: " + mmprojInSameDir);
                        return mmprojInSameDir.toString();
                    }
                } catch (Exception e) {
                    System.err.println("Error searching for mmproj file: " + e.getMessage());
                }
            }

            // Check if this is an integrated LLaVA model (LLaVA-Phi models often have integrated vision)
            if (fullPath.contains("llava-phi")) {
                // For LLaVA-Phi models, vision components might be integrated
                // Return the same model path to indicate integrated vision
                System.err.println("Detected integrated LLaVA-Phi model - using same file for vision projector");
                return languageModelName;
            }

            // Detect which LLaVA model we're using and find appropriate mmproj
            if (baseName.contains("v1.5") || baseName.contains("1.5")) {
                // LLaVA-1.5-7B model
                Path llava15mmproj = Paths.get("models/vlm/llava-1.5-7b-gguf/llava-v1.5-7b-mmproj-model-f16.gguf");
                if (Files.exists(llava15mmproj)) {
                    System.err.println("Using LLaVA-1.5-7B mmproj: " + llava15mmproj);
                    return llava15mmproj.toString();
                }
                
                // Try absolute path
                Path absoluteLlava15 = Paths.get("/home/mythos/work/askmyu/askmyu-forge/java/models/vlm/llava-1.5-7b-gguf/llava-v1.5-7b-mmproj-model-f16.gguf");
                if (Files.exists(absoluteLlava15)) {
                    System.err.println("Using LLaVA-1.5-7B mmproj (absolute): " + absoluteLlava15);
                    return absoluteLlava15.toString();
                }
            } else {
                // LLaVA-Llama3-8B model (original)
                Path mmprojPath = Paths.get("models/vlm/llava-mmproj/llava-llama-3-8b-v1_1-mmproj-f16.gguf");
                if (Files.exists(mmprojPath)) {
                    System.err.println("Using LLaVA-Llama3-8B mmproj: " + mmprojPath);
                    return mmprojPath.toString();
                }
                
                // Try with absolute path
                Path absoluteMMProjPath = Paths.get("/home/mythos/work/askmyu/askmyu-forge/java/models/vlm/llava-mmproj/llava-llama-3-8b-v1_1-mmproj-f16.gguf");
                if (Files.exists(absoluteMMProjPath)) {
                    System.err.println("Using LLaVA-Llama3-8B mmproj (absolute): " + absoluteMMProjPath);
                    return absoluteMMProjPath.toString();
                }
            }
            
            // Final fallback: look in current directory
            System.err.println("Fallback: looking for mmproj in current directory");
            return "llava-llama-3-8b-v1_1-mmproj-f16.gguf";
        }
        
        throw new RuntimeException("Cannot infer vision projector path for model: " + languageModelName);
    }

    /**
     * Load vision projector weights from the mmproj GGUF file.
     * This contains the 377 vision tensors needed for CLIP processing.
     * For integrated models, extracts vision tensors from the main tensor entries.
     */
    private Map<String, GGMLTensorEntry> loadVisionProjector(String visionProjectorPath, Map<String, GGMLTensorEntry> mainTensorEntries) {
        try {
            System.out.println("Loading vision projector from: " + visionProjectorPath);

            // Check if this is an integrated model (vision projector path same as main model)
            if (visionProjectorPath.toLowerCase().contains("llava-phi") && visionProjectorPath.contains("ggml-model")) {
                System.out.println("Detected integrated LLaVA-Phi model - extracting vision tensors from main model");

                // Extract vision-related tensors from the main tensor entries
                Map<String, GGMLTensorEntry> visionTensors = new java.util.HashMap<>();
                for (Map.Entry<String, GGMLTensorEntry> entry : mainTensorEntries.entrySet()) {
                    String tensorName = entry.getKey();
                    // Look for vision-related tensor names (common patterns in LLaVA models)
                    if (tensorName.startsWith("vision.") ||
                        tensorName.startsWith("mm_projector.") ||
                        tensorName.startsWith("vision_encoder.") ||
                        tensorName.startsWith("visual.") ||
                        tensorName.contains("vision") ||
                        tensorName.contains("projector")) {
                        visionTensors.put(tensorName, entry.getValue());
                        System.out.println("Found vision tensor: " + tensorName);
                    }
                }

                if (visionTensors.isEmpty()) {
                    System.err.println("WARNING: No vision tensors found in integrated model - model may not be multimodal");
                    System.err.println("Available tensor prefixes:");
                    mainTensorEntries.keySet().stream()
                        .map(name -> name.split("\\.")[0])
                        .distinct()
                        .sorted()
                        .forEach(prefix -> System.err.println("  " + prefix));
                }

                System.out.println("Extracted " + visionTensors.size() + " vision tensors from integrated model");
                return visionTensors;
            }

            Path projectorPath = Paths.get(visionProjectorPath);
            if (!Files.exists(projectorPath)) {
                // Try relative to current working directory
                projectorPath = Paths.get(System.getProperty("user.dir")).resolve(visionProjectorPath);
            }
            
            if (!Files.exists(projectorPath)) {
                throw new RuntimeException("Vision projector file not found: " + visionProjectorPath);
            }
            
            // Load the mmproj GGUF file
            try (FileChannel projectorChannel = FileChannel.open(projectorPath, java.nio.file.StandardOpenOption.READ)) {
                GGUF projectorGGUF = GGUF.loadModel(projectorPath);
                Map<String, GGMLTensorEntry> visionTensors = GGUF.loadTensors(
                    projectorChannel,
                    projectorGGUF.getTensorDataOffset(),
                    projectorGGUF.getTensorInfos()
                );
                
                System.out.println("Vision projector loaded successfully:");
                System.out.println("  Vision tensors: " + visionTensors.size());
                System.out.println("  File size: " + Files.size(projectorPath) / (1024 * 1024) + " MB");
                
                // Log some key vision tensors to verify they loaded
                logVisionTensors(visionTensors);
                
                return visionTensors;
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load vision projector: " + visionProjectorPath, e);
        }
    }

    /**
     * Log key vision tensors for debugging.
     */
    private void logVisionTensors(Map<String, GGMLTensorEntry> visionTensors) {
        System.out.println("Vision tensor inventory:");
        
        // Count different types of tensors
        int clipTensors = 0;
        int projectorTensors = 0;
        int otherTensors = 0;
        
        for (String tensorName : visionTensors.keySet()) {
            // Recognize v.* tensors as CLIP vision tensors (llama.cpp format)
            if (tensorName.startsWith("vision_model") || tensorName.startsWith("clip") || 
                tensorName.startsWith("v.blk.") || tensorName.startsWith("v.class_embd") || 
                tensorName.startsWith("v.patch_embd") || tensorName.startsWith("v.position_embd") || 
                tensorName.startsWith("v.pre_ln")) {
                clipTensors++;
            } else if (tensorName.startsWith("mm_projector") || tensorName.startsWith("mm.") || tensorName.contains("proj")) {
                projectorTensors++;  
            } else {
                otherTensors++;
            }
        }
        
        System.out.println("  CLIP vision tensors: " + clipTensors);
        System.out.println("  Projector tensors: " + projectorTensors);
        System.out.println("  Other tensors: " + otherTensors);
        System.out.println("  Total: " + visionTensors.size() + " tensors");
        
        // Show first few tensor names as examples
        System.out.println("ALL tensor names for debugging:");
        visionTensors.keySet().stream()
            .sorted()
            .forEach(name -> System.out.println("    " + name));
        
        System.out.println("Sample tensor names:");
        visionTensors.keySet().stream().limit(5).forEach(name -> 
            System.out.println("    " + name + " -> " + visionTensors.get(name).shape())
        );
    }
}