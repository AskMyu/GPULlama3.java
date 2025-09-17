package org.beehive.gpullama3.model.llava;

import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.VLMState;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.vision.encoder.VisionEncoder;
import org.beehive.gpullama3.vision.encoder.ClipVisionEncoder;
import org.beehive.gpullama3.vision.encoder.ClipVisionEncoderGPU;
import org.beehive.gpullama3.vision.projector.MLPProjector;
import org.beehive.gpullama3.vision.projector.MLPProjectorGPU;
import org.beehive.gpullama3.multimodal.data.ImageData;
import org.beehive.gpullama3.multimodal.data.MultimodalInput;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.function.Consumer;

/**
 * LLaVA (Large Language and Vision Assistant) model implementation.
 * Combines CLIP vision encoder with Llama language model through MLP projector.
 * 
 * Architecture:
 * - Vision: CLIP-ViT-Large-patch14-336 (1024-dim features)
 * - Projector: 2-layer MLP (1024 -> 4096)  
 * - Language: Llama-3-8B-Instruct (4096-dim embeddings)
 */
public final class Llava implements Model {
    
    private final LlavaConfiguration config;
    private final Tokenizer tokenizer;
    private final Weights weights;
    private final VisionEncoder visionEncoder;
    private final MLPProjector projector;
    private final boolean useGPUAcceleration;
    private TornadoVMMasterPlan tornadoVMPlan;
    
    public Llava(LlavaConfiguration config, 
                 Tokenizer tokenizer,
                 Map<String, GGMLTensorEntry> languageTensors,
                 Map<String, GGMLTensorEntry> visionTensors,
                 Weights weights) {
        this.config = config;
        this.tokenizer = tokenizer;
        this.weights = weights;
        
        // Determine GPU acceleration availability
        this.useGPUAcceleration = shouldUseGPUAcceleration();
        
        // Initialize vision components with GPU acceleration if available
        if (useGPUAcceleration) {
            System.out.println("[GPU] Initializing GPU-accelerated vision components");
            this.visionEncoder = new ClipVisionEncoderGPU(config, visionTensors, null);
            this.projector = new MLPProjectorGPU(config, visionTensors);
        } else {
            System.out.println("[CPU] Initializing CPU vision components");
            this.visionEncoder = new ClipVisionEncoder(config, visionTensors);
            this.projector = new MLPProjector(config, visionTensors);
        }
        
        System.out.println("LLaVA model initialized: " + config.getModelDescription() + 
                          (useGPUAcceleration ? " with GPU acceleration" : " with CPU processing"));
    }

    @Override
    public ModelType getModelType() {
        return config.isQuantizedInt4() ? ModelType.LLAVA_LLAMA_3_8B_INT4 : ModelType.LLAVA_LLAMA_3_8B;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public Weights weights() {
        return weights;
    }

    @Override
    public LlavaConfiguration configuration() {
        return config;
    }

    @Override
    public ChatFormat chatFormat() {
        // Create a ChatFormat based on the tokenizer type
        return ChatFormat.create(tokenizer, null);
    }

    @Override
    public TornadoVMMasterPlan tornadoVMPlan() {
        return tornadoVMPlan;
    }

    @Override
    public void setTornadoVMPlan(TornadoVMMasterPlan plan) {
        this.tornadoVMPlan = plan;
    }

    @Override
    public State createNewState(int batchsize) {
        // Create a new batched VLM state for LLaVA inference
        VLMState state = new VLMState(config, batchsize);
        
        // Initialize with begin of text token if available
        if (tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        } else {
            // Fallback to BOS token or first token
            state.latestToken = tokenizer.getSpecialTokens().getOrDefault("<|bos|>", 1);
        }
        
        return state;
    }

    public long memoryUsage() {
        // Estimate memory usage for LLaVA model
        long languageModelMemory = config.dim() * config.vocabularySize() * 4; // embeddings
        languageModelMemory += config.numberOfLayers() * config.dim() * config.hiddenDim() * 4; // layers
        
        long visionModelMemory = 1024 * 1024 * 4; // CLIP vision encoder ~4MB
        long projectorMemory = config.getVisionInputDim() * config.getLanguageEmbeddingDim() * 4; // projector
        
        return languageModelMemory + visionModelMemory + projectorMemory;
    }

    @Override
    public void forward(State state, int token, int position) {
        // Standard forward pass for a single token
        // This would delegate to the underlying Llama model
        throw new UnsupportedOperationException("Use multimodal methods for LLaVA model");
    }
    
    @Override
    public State createNewState() {
        // Create a new VLM state for LLaVA inference
        VLMState state = new VLMState(config, -1);
        
        // Initialize with begin of text token if available
        if (tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        } else {
            // Fallback to BOS token or first token
            state.latestToken = tokenizer.getSpecialTokens().getOrDefault("<|bos|>", 1);
        }
        
        return state;
    }
    
    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens,
                                       Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                       boolean echo, IntConsumer onTokenGenerated) {
        return generateTokensWithOptions(state, startPosition, promptTokens, stopTokens, maxTokens,
                                        sampler, echo, onTokenGenerated, null);
    }

    /**
     * Generate tokens with Options for TornadoVM configuration
     */
    public List<Integer> generateTokensWithOptions(State state, int startPosition, List<Integer> promptTokens,
                                                  Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                                  boolean echo, IntConsumer onTokenGenerated, Options options) {
        // TornadoVM plan should already be initialized early in generateTokensMultimodal()
        // No need to initialize here as it would be too late (after vision processing)

        // LLaVA uses Llama as its language backbone, so delegate to Llama CPU generation
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens,
                                                  stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }
    
    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, 
                                          Set<Integer> stopTokens, int maxTokens, Sampler sampler, 
                                          boolean echo, IntConsumer onTokenGenerated, 
                                          TornadoVMMasterPlan tornadoVMPlan) {
        // LLaVA uses Llama as its language backbone, so delegate to Llama GPU generation
        // The multimodal processing (vision + text) happens in preprocessing
        // Once we have text tokens, we can use standard Llama inference
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition, promptTokens, 
                                                     stopTokens, maxTokens, sampler, echo, 
                                                     onTokenGenerated, tornadoVMPlan);
    }

    /**
     * Multimodal forward pass that processes both text and images.
     */
    public FloatArray multimodalForward(MultimodalInput input) {
        int[] textTokens = input.getTextTokens();
        List<ImageData> images = input.getImages();
        
        // Process images through vision encoder
        FloatArray[] visionFeatures = new FloatArray[images.size()];
        for (int i = 0; i < images.size(); i++) {
            // Encode image to CLIP features (1024-dim)
            FloatArray clipFeatures = visionEncoder.encode(images.get(i));
            
            // Project to language model dimensions (4096-dim)
            visionFeatures[i] = projector.project(clipFeatures);
        }
        
        // Combine vision features with text tokens
        FloatArray combinedFeatures = combineVisionAndText(visionFeatures, textTokens);
        
        return combinedFeatures;
    }

    /**
     * Generate tokens with multimodal input (text + images) using direct embedding injection.
     * This method bypasses problematic tokenization by injecting vision embeddings directly into VLMState.
     */
    public List<Integer> generateTokensMultimodal(MultimodalInput input, int maxNewTokens, Sampler sampler, Options options) {
        // Create VLM state for generation
        VLMState vlmState = (VLMState) createNewState();

        // ===== TORNADOVM INITIALIZATION =====
        boolean shouldUseTornadoVM = useGPUAcceleration && options.useTornadovm();

        // Initialize TornadoVM plan if enabled
        if (shouldUseTornadoVM && tornadoVMPlan == null) {
            System.err.println("[LLAVA-MEMORY] Initializing TornadoVM plan for GPU acceleration...");

            // Force GPU memory cleanup before TornadoVM initialization
            forceGPUMemoryCleanup();

            try {
                System.err.println("[LLAVA-MEMORY] Attempting TornadoVM plan initialization...");
                tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(vlmState, this);

                if (tornadoVMPlan != null) {
                    System.err.println("[LLAVA-MEMORY] ✅ TornadoVM plan initialized successfully");
                } else {
                    System.err.println("[LLAVA-MEMORY] ❌ TornadoVM plan initialization returned null");
                    shouldUseTornadoVM = false; // Disable TornadoVM if plan failed to initialize
                }
            } catch (Exception e) {
                System.err.printf("[LLAVA-MEMORY] ❌ TornadoVM plan initialization failed: %s%n", e.getMessage());
                shouldUseTornadoVM = false; // Disable TornadoVM if initialization fails
                tornadoVMPlan = null;
                e.printStackTrace();
            }
        }

        // ===== CRITICAL: WEIGHT/PROCESSING MODE CONSISTENCY CHECK =====
        // Ensure weights type matches processing mode - prevent inconsistent states
        boolean hasGPUWeights = (this.weights() instanceof org.beehive.gpullama3.inference.weights.tornado.TornadoWeights);
        boolean hasValidTornadoPlan = (tornadoVMPlan != null);

        System.err.printf("[LLAVA-MEMORY] ===== CONSISTENCY CHECK =====\n");
        System.err.printf("[LLAVA-MEMORY] GPU Weights loaded: %s\n", hasGPUWeights);
        System.err.printf("[LLAVA-MEMORY] TornadoVM plan valid: %s\n", hasValidTornadoPlan);
        System.err.printf("[LLAVA-MEMORY] shouldUseTornadoVM: %s\n", shouldUseTornadoVM);

        if (hasGPUWeights && !hasValidTornadoPlan) {
            System.err.println("[LLAVA-MEMORY] ❌ CRITICAL: TornadoWeights loaded but no valid TornadoVM plan!");
            System.err.println("[LLAVA-MEMORY] This indicates insufficient GPU memory for full model allocation.");
            System.err.println("[LLAVA-MEMORY] Recommendation: Increase tornado.device.memory or use a smaller model.");
            throw new RuntimeException("Inconsistent state: TornadoWeights without valid TornadoVM execution plan. " +
                                     "GPU memory insufficient for model size. Try increasing -Dtornado.device.memory=<X>GB");
        } else if (!hasGPUWeights && shouldUseTornadoVM) {
            System.err.println("[LLAVA-MEMORY] ⚠️ Warning: TornadoVM requested but StandardWeights loaded - using CPU processing");
            shouldUseTornadoVM = false;
        }
        System.err.printf("[LLAVA-MEMORY] ===== CONSISTENCY CHECK COMPLETE =====\n");

        // Process vision features after TornadoVM is initialized
        List<ImageData> images = input.getImages();
        String textPrompt = input.getCombinedText();
        
        // Get proper stop tokens from chat format for consistent stopping behavior
        Set<Integer> stopTokens = chatFormat().getStopTokens();
        
        if (images.isEmpty()) {
            // No images, fall back to text-only generation
            List<Integer> textTokens = tokenizer().encode(textPrompt, tokenizer().getSpecialTokens().keySet());
            
            // Create token consumer for streaming
            IntConsumer tokenConsumer = null;
            if (options.stream()) {
                tokenConsumer = token -> {
                    if (tokenizer().shouldDisplayToken(token)) {
                        System.out.print(tokenizer().decode(List.of(token)));
                        System.out.flush();
                    }
                };
            }
            
            return generateTokens(vlmState, 0, textTokens, stopTokens, maxNewTokens, sampler, options.echo(), tokenConsumer);
        }
        
        try {
            // Process images through vision pipeline and inject embeddings directly
            long visionStart = System.nanoTime();
            processAndInjectVisionEmbeddings(vlmState, images);
            long visionEnd = System.nanoTime();
            System.err.printf("[PERF] Vision embedding injection took: %.2f ms%n", (visionEnd - visionStart) / 1_000_000.0);
            
            // Create text-only prompt (no vision tokens needed)
            List<Integer> textOnlyPrompt = createTextOnlyPrompt(textPrompt);
            
            // The text tokens start after vision embeddings in the sequence
            int textStartPosition = vlmState.getTextStartPosition();
            System.err.println("[DEBUG] ========== TEXT GENERATION PARAMETERS ==========");
            System.err.println("[DEBUG] VLMState class: " + vlmState.getClass().getSimpleName());
            System.err.println("[DEBUG] Vision tokens injected: " + (vlmState.hasVisionEmbeddings() ? "YES" : "NO"));
            System.err.println("[DEBUG] Text start position: " + textStartPosition);
            System.err.println("[DEBUG] Text prompt tokens: " + textOnlyPrompt);
            System.err.println("[DEBUG] Text prompt size: " + textOnlyPrompt.size());
            System.err.println("[DEBUG] Max new tokens: " + maxNewTokens);
            System.err.println("[DEBUG] Stop tokens: " + stopTokens);
            System.err.println("[DEBUG] ==============================================");
            
            // Force garbage collection before text generation to reduce memory pressure
            System.gc();
            Thread.sleep(100); // Brief pause to allow GC to complete
            
            // Create standard token consumer for streaming
            IntConsumer tokenConsumer = null;
            if (options.stream()) {
                tokenConsumer = token -> {
                    if (tokenizer().shouldDisplayToken(token)) {
                        System.out.print(tokenizer().decode(List.of(token)));
                        System.out.flush();
                    }
                };
            }
            
            // Generate response tokens using standard pipeline with streaming support
            System.err.println("[DEBUG] Starting VLM text generation with streaming...");
            System.err.println("[DEBUG] ===== CRITICAL HANG POINT INVESTIGATION =====");
            System.err.println("[DEBUG] About to call generateTokens with:");
            System.err.println("[DEBUG]   vlmState: " + vlmState.getClass().getSimpleName());
            System.err.println("[DEBUG]   textStartPosition: " + textStartPosition);
            System.err.println("[DEBUG]   textOnlyPrompt size: " + textOnlyPrompt.size());
            System.err.println("[DEBUG]   maxNewTokens: " + maxNewTokens);
            System.err.println("[DEBUG]   sampler: " + sampler.getClass().getSimpleName());
            System.err.println("[DEBUG]   echo: " + options.echo());
            System.err.println("[DEBUG] ===============================================");
            System.err.flush();
            
            long genStart = System.nanoTime();
            
            System.err.println("[DEBUG] CALLING generateTokens NOW...");
            System.err.flush();
            List<Integer> result = generateTokensWithOptions(vlmState, textStartPosition, textOnlyPrompt,
                                                stopTokens, maxNewTokens, sampler, options.echo(), tokenConsumer, options);
            System.err.println("[DEBUG] generateTokens RETURNED successfully!");
            
            long genEnd = System.nanoTime();
            System.err.printf("[PERF] VLM text generation took: %.2f ms%n", (genEnd - genStart) / 1_000_000.0);
            return result;
            
        } catch (Exception e) {
            System.err.println("Error in multimodal generation with direct embedding injection: " + e.getMessage());
            e.printStackTrace();
            
            // Fallback to text-only generation with streaming support
            List<Integer> textTokens = tokenizer().encode(textPrompt, tokenizer().getSpecialTokens().keySet());
            
            IntConsumer fallbackTokenConsumer = null;
            if (options.stream()) {
                fallbackTokenConsumer = token -> {
                    if (tokenizer().shouldDisplayToken(token)) {
                        System.out.print(tokenizer().decode(List.of(token)));
                        System.out.flush();
                    }
                };
            }
            
            return generateTokens(vlmState, 0, textTokens, stopTokens, maxNewTokens, sampler, options.echo(), fallbackTokenConsumer);
        }
    }
    
    /**
     * Process images through vision pipeline and inject embeddings directly into VLMState.
     * This bypasses tokenization entirely and enables true multimodal fusion.
     */
    private void processAndInjectVisionEmbeddings(VLMState vlmState, List<ImageData> images) throws Exception {
        System.err.println("[VLM-DEBUG] ===== ENTERING processAndInjectVisionEmbeddings =====");
        if (images.isEmpty()) {
            System.err.println("[VLM-DEBUG] No images provided, returning early");
            return;
        }
        
        // For simplicity, process the first image (can be extended for multiple images)
        ImageData image = images.get(0);
        System.err.println("[VLM-DEBUG] Processing image: " + image.getWidth() + "x" + image.getHeight());
        
        System.err.println("[VLM-DEBUG] About to call visionEncoder.encode() - THIS MIGHT HANG");
        // Process image through CLIP vision encoder
        FloatArray clipFeatures = visionEncoder.encode(image);
        System.err.println("[VLM-DEBUG] visionEncoder.encode() completed successfully");
        System.err.println("✅ CLIP vision encoding completed: " + clipFeatures.getSize() + " features");
        
        // ===== VISION EMBEDDING QUALITY ANALYSIS ROLLBACK MARKER START =====
        analyzeVisionEmbeddingQuality(clipFeatures, image);
        // ===== VISION EMBEDDING QUALITY ANALYSIS ROLLBACK MARKER END =====
        
        // CRITICAL GPU RESOURCE CLEANUP: Release vision encoder GPU resources before MLP projection
        // This prevents GPU deadlock by ensuring clean GPU context for MLP projector
        System.err.println("[VLM-DEBUG] About to release vision encoder GPU resources...");
        if (visionEncoder instanceof org.beehive.gpullama3.vision.encoder.ClipVisionEncoderGPU) {
            ((org.beehive.gpullama3.vision.encoder.ClipVisionEncoderGPU) visionEncoder).releaseGPUResources();
        }
        System.err.println("[VLM-DEBUG] Vision encoder GPU resources released successfully");
        
        System.err.println("[VLM-DEBUG] About to call projector.project() - THIS MIGHT HANG");
        System.err.println("[VLM-DEBUG] Projector class: " + projector.getClass().getSimpleName());
        System.err.println("[VLM-DEBUG] ClipFeatures size: " + clipFeatures.getSize());
        
        // Project CLIP features - ensure GPU resources are properly synchronized
        FloatArray projectedFeatures;
        try {
            System.err.println("[VLM-DEBUG] Synchronizing GPU resources before projection...");
            
            // FIX: Ensure vision GPU operations complete before MLP projection
            // This prevents GPU resource deadlock between vision encoder and MLP projector
            if (visionEncoder instanceof org.beehive.gpullama3.vision.encoder.ClipVisionEncoderGPU) {
                System.err.println("[VLM-FIX] Ensuring GPU vision encoder completes all operations...");
                // Force synchronization point - vision GPU must finish before MLP GPU starts
                System.gc(); // Hint to release any temporary GPU buffers
                Thread.sleep(100); // Brief pause to ensure GPU context switch
            }
            
            System.err.println("[VLM-DEBUG] CALLING projector.project() on GPU...");
            projectedFeatures = projector.project(clipFeatures);
            System.err.println("[VLM-DEBUG] GPU projector.project() RETURNED successfully");
            
            // ===== MLP PROJECTOR QUALITY ANALYSIS ROLLBACK MARKER START =====
            analyzeMLPProjectorQuality(clipFeatures, projectedFeatures, image);
            // ===== MLP PROJECTOR QUALITY ANALYSIS ROLLBACK MARKER END =====
        } catch (Exception e) {
            System.err.println("[VLM-ERROR] MLP projector failed: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("MLP projector failed: " + e.getMessage(), e);
        }
        System.err.println("[VLM-DEBUG] projector.project() completed successfully");
        System.err.println("✅ MLP projection completed: " + projectedFeatures.getSize() + " projected features");
        
        // Split projected features into individual vision token embeddings
        int embeddingDim = config.dim(); // 4096 for Llama-3-8B
        // Calculate actual number of vision tokens from projectedFeatures (may be reduced by token reduction)
        int actualVisionTokens = projectedFeatures.getSize() / embeddingDim;
        System.err.println("[VLM-DEBUG] Actual vision tokens after reduction: " + actualVisionTokens);
        
        if (projectedFeatures.getSize() % embeddingDim != 0) {
            throw new Exception(String.format("Vision projection size not aligned: size=%d, embeddingDim=%d", 
                                            projectedFeatures.getSize(), embeddingDim));
        }
        
        // Inject vision embeddings into VLM state starting at position 0
        System.err.println("[VLM-DEBUG] About to call vlmState.setVisionEmbeddings() - THIS MIGHT HANG");
        vlmState.setVisionEmbeddings(projectedFeatures, actualVisionTokens, 0);
        System.err.println("[VLM-DEBUG] vlmState.setVisionEmbeddings() completed successfully");
        
        System.err.println("✅ Vision embeddings injected into VLMState: " + actualVisionTokens + " tokens at positions 0-" + (actualVisionTokens - 1));
        System.err.println("✅ Text will start at position: " + vlmState.getTextStartPosition());
        System.err.println("[VLM-DEBUG] ===== EXITING processAndInjectVisionEmbeddings =====");
    }
    
    
    /**
     * Create text-only prompt without vision tokens (since vision is handled by direct embedding injection).
     * Simplified approach to avoid tokenization issues with special tokens.
     */
    private List<Integer> createTextOnlyPrompt(String textPrompt) {
        List<Integer> promptTokens = new ArrayList<>();
        
        System.err.println("[VLM-DEBUG] Starting text tokenization for prompt: '" + textPrompt + "'");
        
        try {
            System.err.println("[VLM-DEBUG] Step 1: Adding begin token...");
            // Add begin of text token if available
            if (tokenizer().getSpecialTokens().containsKey("<|begin_of_text|>")) {
                promptTokens.add(tokenizer().getSpecialTokens().get("<|begin_of_text|>"));
                System.err.println("[VLM-DEBUG] Added begin_of_text token");
            }
            
            System.err.println("[VLM-DEBUG] Step 2: Creating simple prompt...");
            // Create standard LLaVA-1.5 prompt format: "USER: [prompt] ASSISTANT:"
            // Based on official LLaVA conversation template
            String simplePrompt = "USER: " + (textPrompt != null ? textPrompt : "What do you see?") + " ASSISTANT:";
            System.err.println("[VLM-DEBUG] Simple prompt created: '" + simplePrompt + "'");
            
            System.err.println("[VLM-DEBUG] Step 3: About to call tokenizer.encode() - THIS IS WHERE IT MIGHT HANG");
            
            // Try safer tokenization without special tokens first
            List<Integer> textTokens;
            try {
                // First attempt: encode without special tokens to avoid conflicts
                textTokens = tokenizer().encode(simplePrompt, Set.of());
                System.err.println("[VLM-DEBUG] Successfully encoded with no special tokens: " + textTokens.size() + " tokens");
            } catch (Exception tokenizeError) {
                System.err.println("[VLM-DEBUG] Failed to encode without special tokens, trying minimal tokenization");
                // Minimal fallback: just encode the raw text
                textTokens = tokenizer().encode(textPrompt != null ? textPrompt : "Hi", Set.of());
                System.err.println("[VLM-DEBUG] Minimal tokenization successful: " + textTokens.size() + " tokens");
            }
            
            promptTokens.addAll(textTokens);
            System.err.println("[VLM-DEBUG] Text tokenization completed successfully with " + promptTokens.size() + " tokens");
            
        } catch (Exception e) {
            System.err.println("[VLM-ERROR] Error in text tokenization, using fallback approach: " + e.getMessage());
            e.printStackTrace();
            
            // Ultimate fallback: just encode the text prompt directly
            if (textPrompt != null && !textPrompt.isEmpty()) {
                try {
                    List<Integer> fallbackTokens = tokenizer().encode(textPrompt, Set.of());
                    promptTokens.addAll(fallbackTokens);
                    System.err.println("[VLM-DEBUG] Fallback tokenization successful: " + fallbackTokens.size() + " tokens");
                } catch (Exception e2) {
                    System.err.println("[VLM-ERROR] Could not tokenize text even with fallback: " + e2.getMessage());
                    // Add a single dummy token as absolute fallback
                    promptTokens.add(1); // BOS token or similar
                    System.err.println("[VLM-DEBUG] Using dummy token fallback");
                }
            } else {
                // Add a single dummy token if no text provided
                promptTokens.add(1);
                System.err.println("[VLM-DEBUG] Using dummy token for empty prompt");
            }
        }
        
        return promptTokens;
    }

    /**
     * Create multimodal prompt tokens that include vision features.
     * NOTE: This method is now deprecated in favor of direct embedding injection.
     * Kept for reference and potential fallback scenarios.
     */
    @Deprecated
    private List<Integer> createMultimodalPrompt(List<ImageData> images, String textPrompt) {
        List<Integer> promptTokens = new ArrayList<>();
        
        // Add begin of text token
        if (tokenizer().getSpecialTokens().containsKey("<|begin_of_text|>")) {
            promptTokens.add(tokenizer().getSpecialTokens().get("<|begin_of_text|>"));
        }
        
        // Start user message
        if (tokenizer().getSpecialTokens().containsKey("<|start_header_id|>")) {
            promptTokens.add(tokenizer().getSpecialTokens().get("<|start_header_id|>"));
            promptTokens.addAll(tokenizer().encode("user", tokenizer().getSpecialTokens().keySet()));
            promptTokens.add(tokenizer().getSpecialTokens().get("<|end_header_id|>"));
        }
        
        // Process each image - for LLaVA, we represent images with placeholder tokens
        for (ImageData image : images) {
            // Process image through vision pipeline 
            try {
                FloatArray clipFeatures = visionEncoder.encode(image);
                FloatArray visionEmbeddings = projector.project(clipFeatures);
                
                // For LLaVA, we add a special marker followed by vision placeholder tokens
                // In a complete implementation, these would be handled at the embedding level
                promptTokens.addAll(tokenizer().encode("<image>", tokenizer().getSpecialTokens().keySet()));
                
                // Add vision placeholder tokens (576 tokens for 24x24 patches)
                // These would be replaced with actual vision embeddings in the forward pass
                for (int i = 0; i < config.getVisionTokenCount(); i++) {
                    // Use a range of unused token IDs (above vocabulary size)
                    promptTokens.add(config.vocabularySize() + i);
                }
                
                promptTokens.addAll(tokenizer().encode("</image>", tokenizer().getSpecialTokens().keySet()));
                
            } catch (Exception e) {
                System.err.println("Error processing image in multimodal prompt: " + e.getMessage());
                // Add fallback text representation
                promptTokens.addAll(tokenizer().encode("[IMAGE]", tokenizer().getSpecialTokens().keySet()));
            }
        }
        
        // Add text prompt
        if (textPrompt != null && !textPrompt.isEmpty()) {
            // Add space and text prompt together to avoid tokenization issues
            String spacedPrompt = " " + textPrompt;
            List<Integer> textTokens = tokenizer().encode(spacedPrompt, tokenizer().getSpecialTokens().keySet());
            promptTokens.addAll(textTokens);
        }
        
        // Start assistant response
        if (tokenizer().getSpecialTokens().containsKey("<|start_header_id|>")) {
            promptTokens.add(tokenizer().getSpecialTokens().get("<|start_header_id|>"));
            promptTokens.addAll(tokenizer().encode("assistant", tokenizer().getSpecialTokens().keySet()));
            promptTokens.add(tokenizer().getSpecialTokens().get("<|end_header_id|>"));
        }
        
        return promptTokens;
    }

    // Remove these methods as they don't match the Model interface

    /**
     * Combine vision features with text tokens for unified processing.
     */
    private FloatArray combineVisionAndText(FloatArray[] visionFeatures, int[] textTokens) {
        // This is a simplified implementation
        // Full version would properly interleave vision tokens with text tokens
        // according to the LLaVA conversation format
        
        int totalVisionTokens = visionFeatures.length * config.getVisionTokenCount();
        int totalTextTokens = textTokens != null ? textTokens.length : 0;
        int totalTokens = totalVisionTokens + totalTextTokens;
        
        FloatArray combined = new FloatArray(totalTokens * config.dim());
        
        // Simplified implementation: just add vision features first, then text
        int offset = 0;
        
        // Add vision tokens
        for (int i = 0; i < visionFeatures.length; i++) {
            for (int j = 0; j < visionFeatures[i].getSize(); j++) {
                combined.set(offset + j, visionFeatures[i].get(j));
            }
            offset += visionFeatures[i].getSize();
        }
        
        // Add text tokens (would need proper token embedding lookup)
        if (textTokens != null) {
            for (int tokenIdx = 0; tokenIdx < textTokens.length; tokenIdx++) {
                // This would need proper embedding lookup from the language model
                // For now, just placeholder values
                for (int dim = 0; dim < config.dim(); dim++) {
                    combined.set(offset + tokenIdx * config.dim() + dim, 0.0f);
                }
            }
        }
        
        return combined;
    }

    /**
     * Check if this model supports vision processing.
     */
    public boolean supportsVision() {
        return true;
    }
    
    /**
     * Determine if GPU acceleration should be used based on TornadoVM availability.
     */
    /**
     * Force GPU memory cleanup before TornadoVM initialization to prevent fragmentation.
     */
    private void forceGPUMemoryCleanup() {
        System.err.println("[GPU-CLEANUP] ===== FORCING GPU MEMORY CLEANUP =====");

        // 1. Clean up vision encoder GPU resources
        if (visionEncoder instanceof ClipVisionEncoderGPU) {
            System.err.println("[GPU-CLEANUP] Cleaning up vision encoder GPU resources...");
            try {
                ((ClipVisionEncoderGPU) visionEncoder).close();
                System.err.println("[GPU-CLEANUP] ✅ Vision encoder GPU resources cleaned up");
            } catch (Exception e) {
                System.err.println("[GPU-CLEANUP] Warning: Vision encoder cleanup failed: " + e.getMessage());
            }
        }

        // 2. Clean up projector GPU resources
        if (projector instanceof MLPProjectorGPU) {
            System.err.println("[GPU-CLEANUP] Cleaning up projector GPU resources...");
            try {
                ((MLPProjectorGPU) projector).close();
                System.err.println("[GPU-CLEANUP] ✅ Projector GPU resources cleaned up");
            } catch (Exception e) {
                System.err.println("[GPU-CLEANUP] Warning: Projector cleanup failed: " + e.getMessage());
            }
        }

        // 3. Force Java garbage collection to release any dangling GPU references
        System.err.println("[GPU-CLEANUP] Forcing Java garbage collection...");
        System.gc();
        System.runFinalization();

        // 4. Brief delay to allow GPU driver to process cleanup
        try {
            Thread.sleep(200); // Slightly longer delay for thorough cleanup
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // 5. Display memory status
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory() / (1024 * 1024);
        long freeMemory = runtime.freeMemory() / (1024 * 1024);
        long usedMemory = totalMemory - freeMemory;
        System.err.printf("[GPU-CLEANUP] JVM Memory: Used %d MB / Total %d MB (%.1f%% used)%n",
                         usedMemory, totalMemory, (usedMemory * 100.0) / totalMemory);

        System.err.println("[GPU-CLEANUP] ===== GPU MEMORY CLEANUP COMPLETE =====");
    }

    private boolean shouldUseGPUAcceleration() {
        try {
            // Check if TornadoVM classes exist without initializing them
            // Use reflection to avoid static initialization
            ClassLoader cl = getClass().getClassLoader();
            
            // First check if the basic classes exist
            cl.loadClass("uk.ac.manchester.tornado.api.types.arrays.FloatArray");
            System.out.println("[GPU] TornadoVM FloatArray class found");
            
            // Try to create a FloatArray - this should work even if GPU hardware isn't available
            FloatArray testArray = new FloatArray(5);
            testArray.init(1.0f);
            System.out.println("[GPU] TornadoVM FloatArray creation successful");
            
            // Check for TaskGraph class without initializing the class
            cl.loadClass("uk.ac.manchester.tornado.api.TaskGraph");  
            System.out.println("[GPU] TornadoVM TaskGraph class found");
            
            // Don't try to create TaskGraph here - that's what's causing the crash
            // Let the individual GPU components try that and handle failures
            
            System.out.println("[GPU] TornadoVM classes available - enabling GPU acceleration");
            return true;
            
        } catch (ClassNotFoundException e) {
            System.out.println("[GPU] TornadoVM classes not found: " + e.getMessage());
            return false;
        } catch (Throwable e) {
            System.out.println("[GPU] TornadoVM basic test failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return false;
        }
    }

    /**
     * Get vision encoder for direct access.
     */
    public VisionEncoder getVisionEncoder() {
        return visionEncoder;
    }

    /**
     * Get MLP projector for direct access.
     */
    public MLPProjector getProjector() {
        return projector;
    }
    
    /**
     * Generate multimodal response with streaming callback for external applications.
     * Provides real-time token-by-token streaming with custom callbacks.
     * 
     * @param input the multimodal input containing text and images
     * @param maxTokens maximum number of tokens to generate
     * @param sampler the sampling strategy to use
     * @param onToken callback for each generated token (receives decoded string)
     * @param onComplete callback when generation is complete (receives full response)
     */
    public void generateTokensMultimodalStreaming(MultimodalInput input, int maxTokens, Sampler sampler,
                                                  Consumer<String> onToken, Consumer<String> onComplete) {
        try {
            // Process vision embeddings
            VLMState vlmState = (VLMState) createNewState();
            processAndInjectVisionEmbeddings(vlmState, input.getImages());
            
            List<Integer> textOnlyPrompt = createTextOnlyPrompt(input.getCombinedText());
            Set<Integer> stopTokens = chatFormat().getStopTokens();
            int textStartPosition = vlmState.getTextStartPosition();
            
            // Create token consumer for real-time streaming
            IntConsumer tokenConsumer = token -> {
                if (tokenizer().shouldDisplayToken(token)) {
                    String decoded = tokenizer().decode(List.of(token));
                    onToken.accept(decoded);
                }
            };
            
            // Generate with streaming
            List<Integer> tokens = generateTokens(vlmState, textStartPosition, textOnlyPrompt,
                                                stopTokens, maxTokens, sampler, false, tokenConsumer);
            
            // Complete callback with full response
            String fullResponse = tokenizer().decode(tokens);
            onComplete.accept(fullResponse);
            
        } catch (Exception e) {
            onComplete.accept("Error: " + e.getMessage());
        }
    }
    
    /**
     * Generate multimodal response with streaming and status callbacks for external applications.
     * Provides real-time token-by-token streaming with progress updates.
     * 
     * @param input the multimodal input containing text and images
     * @param maxTokens maximum number of tokens to generate
     * @param sampler the sampling strategy to use
     * @param onToken callback for each generated token (receives decoded string)
     * @param onStatus callback for status updates during processing
     * @param onComplete callback when generation is complete (receives full response)
     */
    public void generateTokensMultimodalStreamingWithStatus(MultimodalInput input, int maxTokens, 
                                                           Sampler sampler, Consumer<String> onToken,
                                                           Consumer<String> onStatus, Consumer<String> onComplete) {
        try {
            onStatus.accept("Processing vision features...");
            // Vision processing with progress updates
            VLMState vlmState = (VLMState) createNewState();
            processAndInjectVisionEmbeddings(vlmState, input.getImages());
            
            onStatus.accept("Starting text generation...");
            List<Integer> textOnlyPrompt = createTextOnlyPrompt(input.getCombinedText());
            Set<Integer> stopTokens = chatFormat().getStopTokens();
            int textStartPosition = vlmState.getTextStartPosition();
            
            // Create token consumer for real-time streaming
            IntConsumer tokenConsumer = token -> {
                if (tokenizer().shouldDisplayToken(token)) {
                    String decoded = tokenizer().decode(List.of(token));
                    onToken.accept(decoded);
                }
            };
            
            // Generate with streaming
            List<Integer> tokens = generateTokens(vlmState, textStartPosition, textOnlyPrompt,
                                                stopTokens, maxTokens, sampler, false, tokenConsumer);
            
            onStatus.accept("Generation complete");
            String fullResponse = tokenizer().decode(tokens);
            onComplete.accept(fullResponse);
            
        } catch (Exception e) {
            onStatus.accept("Error: " + e.getMessage());
            onComplete.accept("");
        }
    }

    @Override
    public String toString() {
        return String.format("LLaVA[%s, vision_tokens=%d, lang_params=%dB]", 
                           config.getVisionEncoderType(),
                           config.getVisionTokenCount(),
                           config.numberOfLayers() * config.dim() / 1_000_000);
    }
    
    // ===== VISION EMBEDDING QUALITY ANALYSIS ROLLBACK MARKER START =====
    /**
     * Analyze the semantic quality of CLIP vision embeddings
     */
    private static void analyzeVisionEmbeddingQuality(FloatArray clipFeatures, ImageData image) {
        System.err.printf("[🔍 VISION-EMBEDDING-ANALYSIS] Analyzing CLIP features for %dx%d image%n", 
                         image.getWidth(), image.getHeight());
        
        // Basic statistics
        int size = clipFeatures.getSize();
        float sum = 0f, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        int zeroCount = 0, nanCount = 0, infCount = 0;
        
        for (int i = 0; i < size; i++) {
            float val = clipFeatures.get(i);
            if (Float.isNaN(val)) nanCount++;
            else if (Float.isInfinite(val)) infCount++;
            else if (val == 0f) zeroCount++;
            else {
                sum += val;
                min = Math.min(min, val);
                max = Math.max(max, val);
            }
        }
        
        float mean = sum / (size - zeroCount - nanCount - infCount);
        
        System.err.printf("[VISION-STATS] CLIP embeddings: size=%d, mean=%.6f, range=[%.6f,%.6f]%n", 
                         size, mean, min, max);
        System.err.printf("[VISION-QUALITY] Zeros=%d, NaNs=%d, Infs=%d, Valid=%d (%.1f%%)%n", 
                         zeroCount, nanCount, infCount, size - zeroCount - nanCount - infCount,
                         100f * (size - zeroCount - nanCount - infCount) / size);
        
        // Semantic analysis - check if embeddings contain meaningful patterns
        float[] first10 = new float[Math.min(10, size)];
        float[] last10 = new float[Math.min(10, size)];
        for (int i = 0; i < Math.min(10, size); i++) {
            first10[i] = clipFeatures.get(i);
            last10[i] = clipFeatures.get(size - 1 - i);
        }
        
        System.err.print("[VISION-SEMANTIC] First 10 values: ");
        for (float val : first10) {
            System.err.printf("%.4f ", val);
        }
        System.err.println();
        
        System.err.print("[VISION-SEMANTIC] Last 10 values: ");
        for (float val : last10) {
            System.err.printf("%.4f ", val);
        }
        System.err.println();
        
        // Check for uniform distribution (might indicate poor encoding)
        boolean isUniform = true;
        float firstVal = clipFeatures.get(0);
        for (int i = 1; i < Math.min(50, size); i++) {
            if (Math.abs(clipFeatures.get(i) - firstVal) > 0.001f) {
                isUniform = false;
                break;
            }
        }
        
        if (isUniform) {
            System.err.printf("[🚨 VISION-DIAGNOSIS] CRITICAL: Embeddings appear uniform (%.6f), poor semantic extraction!%n", firstVal);
        } else if (zeroCount > size * 0.8f) {
            System.err.printf("[⚠️ VISION-DIAGNOSIS] WARNING: %d%% zeros, sparse representation%n", 100 * zeroCount / size);
        } else if (Math.abs(mean) < 0.001f && max - min < 0.01f) {
            System.err.println("[⚠️ VISION-DIAGNOSIS] WARNING: Very small values, weak semantic signal");
        } else {
            System.err.println("[✅ VISION-DIAGNOSIS] GOOD: Embeddings show meaningful variation and scale");
        }
    }
    
    /**
     * Analyze the quality of MLP projector transformation  
     */
    private static void analyzeMLPProjectorQuality(FloatArray clipFeatures, FloatArray projectedFeatures, ImageData image) {
        System.err.printf("[🔍 MLP-PROJECTOR-ANALYSIS] Analyzing projection: %d → %d dimensions%n", 
                         clipFeatures.getSize(), projectedFeatures.getSize());
        
        // Input analysis
        float clipSum = 0f, clipMin = Float.MAX_VALUE, clipMax = Float.MIN_VALUE;
        for (int i = 0; i < clipFeatures.getSize(); i++) {
            float val = clipFeatures.get(i);
            clipSum += val;
            clipMin = Math.min(clipMin, val);
            clipMax = Math.max(clipMax, val);
        }
        float clipMean = clipSum / clipFeatures.getSize();
        
        // Output analysis  
        float projSum = 0f, projMin = Float.MAX_VALUE, projMax = Float.MIN_VALUE;
        int projZeros = 0, projNaNs = 0;
        for (int i = 0; i < projectedFeatures.getSize(); i++) {
            float val = projectedFeatures.get(i);
            if (Float.isNaN(val)) projNaNs++;
            else if (val == 0f) projZeros++;
            else {
                projSum += val;
                projMin = Math.min(projMin, val);
                projMax = Math.max(projMax, val);
            }
        }
        float projMean = projSum / (projectedFeatures.getSize() - projZeros - projNaNs);
        
        System.err.printf("[MLP-INPUT] CLIP: mean=%.6f, range=[%.6f,%.6f]%n", clipMean, clipMin, clipMax);
        System.err.printf("[MLP-OUTPUT] Projected: mean=%.6f, range=[%.6f,%.6f], zeros=%d, NaNs=%d%n", 
                         projMean, projMin, projMax, projZeros, projNaNs);
        
        // Scale and transformation analysis
        float scaleRatio = (clipMax - clipMin) / (projMax - projMin);
        float meanShift = projMean - clipMean;
        
        System.err.printf("[MLP-TRANSFORM] Scale ratio=%.6f, Mean shift=%.6f%n", scaleRatio, meanShift);
        
        // Sample first and last values for pattern detection
        System.err.print("[MLP-SAMPLE] Input first 5: ");
        for (int i = 0; i < Math.min(5, clipFeatures.getSize()); i++) {
            System.err.printf("%.4f ", clipFeatures.get(i));
        }
        System.err.println();
        
        System.err.print("[MLP-SAMPLE] Output first 5: ");
        for (int i = 0; i < Math.min(5, projectedFeatures.getSize()); i++) {
            System.err.printf("%.4f ", projectedFeatures.get(i));
        }
        System.err.println();
        
        // Quality diagnosis
        if (projNaNs > 0) {
            System.err.printf("[🚨 MLP-DIAGNOSIS] CRITICAL: %d NaN values in projection output!%n", projNaNs);
        } else if (projZeros > projectedFeatures.getSize() * 0.9f) {
            System.err.printf("[🚨 MLP-DIAGNOSIS] CRITICAL: %d%% zeros, projection likely failed!%n", 100 * projZeros / projectedFeatures.getSize());
        } else if (Math.abs(projMean) < 0.001f && projMax - projMin < 0.01f) {
            System.err.println("[⚠️ MLP-DIAGNOSIS] WARNING: Projection output very small, weak transformation");
        } else if (Math.abs(scaleRatio - 1.0f) > 100f) {
            System.err.printf("[⚠️ MLP-DIAGNOSIS] WARNING: Extreme scale change (%.1fx), possible over/under-scaling%n", scaleRatio);
        } else {
            System.err.println("[✅ MLP-DIAGNOSIS] GOOD: Projection shows reasonable transformation");
        }
    }
    // ===== VISION EMBEDDING QUALITY ANALYSIS ROLLBACK MARKER END =====
}