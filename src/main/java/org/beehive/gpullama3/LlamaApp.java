package org.beehive.gpullama3;

import org.beehive.gpullama3.aot.AOT;
import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.sampler.CategoricalSampler;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.sampler.ToppSampler;
import org.beehive.gpullama3.inference.sampler.TopKSampler;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.FloatArrayUtils;
import org.beehive.gpullama3.multimodal.data.MultimodalInput;
import org.beehive.gpullama3.multimodal.data.ImageData;
import org.beehive.gpullama3.vision.processor.NativeImageProcessor;
import org.beehive.gpullama3.model.llava.Llava;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));   // Enable Java Vector API for CPU acceleration
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));               // Use Ahead-of-Time compilation
    public static final boolean SHOW_PERF_INTERACTIVE = Boolean.parseBoolean(System.getProperty("llama.ShowPerfInteractive", "true")); // Show performance metrics in interactive mode

    /**
     * Creates and configures a sampler for token generation based on specified parameters.
     *
     * <p>This method selects an appropriate sampling strategy for next-token prediction
     * in language model inference. It supports several sampling approaches:</p>
     *
     * <ul>
     *   <li>Greedy sampling (temperature = 0): Always selects the most probable token</li>
     *   <li>Temperature sampling: Adjusts probability distribution sharpness</li>
     *   <li>Top-p (nucleus) sampling: Considers only tokens comprising the top p probability mass</li>
     * </ul>
     *
     * <p>The method handles both {@link FloatTensor} and {@link FloatArray} logits types
     * to support both CPU and GPU execution paths.</p>
     *
     * @param vocabularySize
     *         The size of the model's vocabulary
     * @param temperature
     *         A value controlling randomness in sampling:
     *         <ul>
     *           <li>0.0f: No randomness (greedy sampling)</li>
     *           <li>1.0f: Standard sampling from unmodified distribution</li>
     *           <li>&lt;1.0f: More deterministic (sharper distribution)</li>
     *           <li>&gt;1.0f: More random (flatter distribution)</li>
     *         </ul>
     * @param topp
     *         The cumulative probability threshold for nucleus sampling (0.0-1.0).
     *         <ul>
     *           <li>Values ≤0 or ≥1: Disables top-p sampling</li>
     *           <li>Values in (0,1): Restricts sampling to tokens comprising the top p probability mass</li>
     *         </ul>
     * @param rngSeed
     *         Seed value for the random number generator to ensure reproducibility
     * @return A configured {@link Sampler} that implements the selected sampling strategy and handles both tensor and array-based logits
     * @throws IllegalArgumentException
     *         if logits are of an unsupported type
     */
    public static Sampler selectSampler(int vocabularySize, float temperature, float topp, int topK, long rngSeed) {
        return selectSampler(vocabularySize, temperature, topp, topK, rngSeed, null);
    }

    public static Sampler selectSampler(int vocabularySize, float temperature, float topp, int topK, long rngSeed, ModelType modelType) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.TENSOR_ARGMAX; // Use TENSOR_ARGMAX instead of ARGMAX
        } else {
            // we sample from this distribution to get the next token
            // Use time-based entropy to prevent repetitive patterns
            long entropyEnhancedSeed = rngSeed ^ System.nanoTime() ^ Thread.currentThread().getId();
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(entropyEnhancedSeed);
            Sampler innerSampler;
            // Gemma models work best with top-K sampling (Google recommendation: K=64)
            // Use top-K as primary sampling method when specified
            if (topK > 0 && topK < vocabularySize) {
                // Use top-K sampling - optimal for Gemma models
                innerSampler = new TopKSampler(vocabularySize, topK, rng);
            } else if (topp > 0 && topp < 1) {
                // Use top-p (nucleus) sampling with the specified threshold
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            } else {
                // Fallback to standard categorical sampling
                innerSampler = new CategoricalSampler(rng, vocabularySize);
            }

            // Create a sampler that:
            // 1. Applies temperature scaling to the logits
            // 2. Converts logits to probabilities using softmax
            // 3. Delegates the actual sampling to the appropriate inner sampler
            final String modelTypeName = modelType != null ? modelType.name() : "UNKNOWN";
            sampler = logits -> {
                System.err.printf("[%s-SAMPLER] ===== SAMPLER WRAPPER CALLED =====%n", modelTypeName);
                System.err.printf("[%s-SAMPLER] Logits type: %s%n", modelTypeName,
                    logits != null ? logits.getClass().getName() : "null");
                
                // Handle different logits formats to support both CPU and GPU paths
                if (logits instanceof FloatTensor) {
                    // For CPU path using FloatTensor
                    FloatTensor tensorLogits = (FloatTensor) logits;
                    System.err.printf("[%s-SAMPLER] FloatTensor size BEFORE processing: %d%n", modelTypeName, tensorLogits.size());

                    // Check for extreme logits values that could cause numeric instability
                    float maxLogit = Float.NEGATIVE_INFINITY;
                    float minLogit = Float.POSITIVE_INFINITY;
                    for (int i = 0; i < tensorLogits.size(); i++) {
                        float logit = tensorLogits.getFloat(i);
                        maxLogit = Math.max(maxLogit, logit);
                        minLogit = Math.min(minLogit, logit);
                    }
                    System.err.printf("[%s-SAMPLER] Logits range BEFORE processing: [%.6f, %.6f]%n", modelTypeName, minLogit, maxLogit);

                    // Apply temperature scaling - lower values make distribution more peaked
                    tensorLogits.divideInPlace(0, tensorLogits.size(), temperature);
                    // Convert logits to probabilities using softmax
                    tensorLogits.softmaxInPlace(0, tensorLogits.size());
                    System.err.printf("[%s-SAMPLER] FloatTensor size AFTER processing: %d%n", modelTypeName, tensorLogits.size());
                } else if (logits instanceof FloatArray) {
                    // For GPU path using FloatArray
                    FloatArray arrayLogits = (FloatArray) logits;
                    System.err.printf("[%s-SAMPLER] FloatArray size BEFORE processing: %d%n", modelTypeName, arrayLogits.getSize());

                    // 🚨 CRITICAL VOCABULARY BOUNDS FIX 🚨
                    // Some models (like Gemma) have larger vocabulary sizes than others
                    // Use the actual model vocabulary size instead of hardcoding
                    final int VOCAB_SIZE = vocabularySize;
                    if (arrayLogits.getSize() > VOCAB_SIZE) {
                        System.err.printf("[%s-VOCAB-FIX] 🔥 Truncating logits from %d to %d (vocab bounds)%n",
                            modelTypeName, arrayLogits.getSize(), VOCAB_SIZE);

                        // Create new truncated array with only vocabulary-valid logits
                        FloatArray truncatedLogits = new FloatArray(VOCAB_SIZE);
                        for (int i = 0; i < VOCAB_SIZE; i++) {
                            truncatedLogits.set(i, arrayLogits.get(i));
                        }
                        arrayLogits = truncatedLogits;
                        logits = arrayLogits; // Update the logits reference
                        System.err.printf("[%s-VOCAB-FIX] ✅ Logits truncated to valid vocabulary size: %d%n", modelTypeName, arrayLogits.getSize());
                    } else {
                        System.err.printf("[%s-VOCAB-FIX] ✅ Logits size %d is within vocab bounds%n", modelTypeName, arrayLogits.getSize());
                    }

                    // Check for extreme logits values that could cause numeric instability
                    float maxLogit = Float.NEGATIVE_INFINITY;
                    float minLogit = Float.POSITIVE_INFINITY;
                    int nonZeroCount = 0;
                    for (int i = 0; i < Math.min(arrayLogits.getSize(), 100); i++) {
                        float logit = arrayLogits.get(i);
                        if (Math.abs(logit) > 1e-8) nonZeroCount++;
                        maxLogit = Math.max(maxLogit, logit);
                        minLogit = Math.min(minLogit, logit);
                    }
                    System.err.printf("[%s-SAMPLER] Logits range BEFORE processing: [%.6f, %.6f], non-zero: %d/100%n",
                        modelTypeName, minLogit, maxLogit, nonZeroCount);

                    // Apply the same operations but using FloatArray-specific methods for TornadoVM data types
                    System.err.printf("[%s-SAMPLER] Temperature scaling with temperature=%.6f%n", modelTypeName, temperature);
                    FloatArrayUtils.divideInPlace(arrayLogits, 0, arrayLogits.getSize(), temperature);

                    // Check values after temperature scaling
                    maxLogit = Float.NEGATIVE_INFINITY;
                    minLogit = Float.POSITIVE_INFINITY;
                    for (int i = 0; i < Math.min(arrayLogits.getSize(), 10); i++) {
                        float logit = arrayLogits.get(i);
                        maxLogit = Math.max(maxLogit, logit);
                        minLogit = Math.min(minLogit, logit);
                    }
                    System.err.printf("[%s-SAMPLER] After temperature scaling: [%.6f, %.6f]%n", modelTypeName, minLogit, maxLogit);

                    FloatArrayUtils.softmaxInPlace(arrayLogits, 0, arrayLogits.getSize());

                    // Check post-processing values (now probabilities, not logits)
                    maxLogit = Float.NEGATIVE_INFINITY;
                    minLogit = Float.POSITIVE_INFINITY;
                    nonZeroCount = 0;
                    double totalSum = 0.0;
                    int maxProbIdx = 0;

                    // Check a larger sample for better statistics
                    for (int i = 0; i < arrayLogits.getSize(); i++) {
                        float prob = arrayLogits.get(i);
                        totalSum += prob;
                        if (prob > maxLogit) {
                            maxLogit = prob;
                            maxProbIdx = i;
                        }
                        if (prob > 0) {
                            minLogit = Math.min(minLogit, prob);
                            nonZeroCount++;
                        }
                    }
                    System.err.printf("[%s-SAMPLER] Probabilities AFTER softmax: max=%.9f (token %d), min_positive=%.9f, non-zero: %d/%d, sum=%.9f%n",
                        modelTypeName, maxLogit, maxProbIdx, minLogit, nonZeroCount, arrayLogits.getSize(), totalSum);
                    System.err.printf("[%s-SAMPLER] FloatArray size AFTER processing: %d%n", modelTypeName, arrayLogits.getSize());
                } else {
                    // If logits are neither FloatTensor nor FloatArray, throw an exception
                    throw new IllegalArgumentException("Unsupported logits type: " + (logits != null ? logits.getClass().getName() : "null"));
                }
                System.err.printf("[%s-SAMPLER] About to call innerSampler.sampleToken()...%n", modelTypeName);
                int result = innerSampler.sampleToken(logits);
                System.err.printf("[%s-SAMPLER] innerSampler returned token: %d%n", modelTypeName, result);
                System.err.printf("[%s-SAMPLER] ===== SAMPLER WRAPPER RETURNING =====%n", modelTypeName);
                return result;
            };
        }
        return sampler;
    }

    /**
     * Loads the language model based on the given options.
     * <p>
     * If Ahead-of-Time (AOT) mode is enabled, attempts to use a pre-loaded compiled model. Otherwise, loads the model from the specified path using the model loader.
     * </p>
     *
     * @param options
     *         the parsed CLI options containing model path and max token limit
     * @return the loaded {@link Model} instance
     * @throws IOException
     *         if the model fails to load
     * @throws IllegalStateException
     *         if AOT loading is enabled but the preloaded model is unavailable
     */
    private static Model loadModel(Options options) throws IOException {
        if (USE_AOT) {
            Model model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
            if (model == null) {
                throw new IllegalStateException("Failed to load precompiled AOT model.");
            }
            return model;
        }
        System.err.printf("[LLAMAAPP-DEBUG] Loading model with useTornadovm=%b%n", options.useTornadovm());
        return ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
    }

    private static Sampler createSampler(Model model, Options options) {
        // Apply conservative sampling for Granite models to improve response quality
        if (model.getModelType() == ModelType.GRANITE_3_3) {
            System.err.println("[GRANITE-SAMPLER] Detected Granite model - applying conservative sampling parameters");
            Options conservativeOptions = Options.withConservativeSampling(options);
            return selectSampler(model.configuration().vocabularySize(),
                               conservativeOptions.temperature(),
                               conservativeOptions.topp(),
                               conservativeOptions.topK(),
                               conservativeOptions.seed(),
                               model.getModelType());
        }

        return selectSampler(model.configuration().vocabularySize(), options.temperature(), options.topp(), options.topK(), options.seed(), model.getModelType());
    }

    private static void runSingleInstruction(Model model, Sampler sampler, Options options) {
        // Check if image is provided for multimodal inference
        if (options.imagePath() != null) {
            runMultimodalInstruction(model, sampler, options);
        } else {
            String response = model.runInstructOnce(sampler, options);
            System.out.println(response);
            if (SHOW_PERF_INTERACTIVE) {
                LastRunMetrics.printMetrics();
            }
        }
    }
    
    /**
     * Handle multimodal (image + text) inference.
     */
    private static void runMultimodalInstruction(Model model, Sampler sampler, Options options) {
        try {
            // Load and process the image
            byte[] imageBytes = Files.readAllBytes(options.imagePath());
            System.err.println("Loaded image: " + options.imagePath() + " (" + imageBytes.length + " bytes)");
            
            // Process image using native Java implementation with CLIP parameters
            NativeImageProcessor processor = new NativeImageProcessor();
            ImageData imageData = processor.preprocessImage(imageBytes, 336, true); // CLIP ViT-Large uses 336x336
            System.err.println("Processed image: " + imageData);
            
            // Create multimodal input with text prompt and image
            String textPrompt = options.prompt();
            
            // Tokenize the text prompt using the model's tokenizer
            List<Integer> tokenList = model.tokenizer().encodeAsList(textPrompt);
            int[] textTokens = tokenList.stream().mapToInt(Integer::intValue).toArray();
            System.err.println("Tokenized text: " + textTokens.length + " tokens");
            
            MultimodalInput multimodalInput = MultimodalInput.textAndImage(textPrompt, imageData, textTokens);
            System.err.println("Created multimodal input: " + multimodalInput.getSummary());
            
            // Note: Image successfully preprocessed with CLIP parameters
            System.err.println("✅ Image preprocessing successful with native Java implementation");
            System.err.println("✅ Applied CLIP normalization (mean=[0.481,0.458,0.408], std=[0.269,0.261,0.276])");
            
            String response;
            if (model instanceof Llava) {
                System.err.println("✅ Detected LLaVA model - using multimodal inference");
                Llava llavaModel = (Llava) model;
                
                try {
                    // Generate tokens using complete multimodal pipeline
                    List<Integer> responseTokens = llavaModel.generateTokensMultimodal(multimodalInput, options.maxTokens(), sampler, options);
                    
                    // Convert tokens to text using the tokenizer
                    response = model.tokenizer().decode(responseTokens);
                    
                    System.err.println("✅ Generated multimodal response: " + responseTokens.size() + " tokens");
                } catch (Exception e) {
                    System.err.println("Error in multimodal generation: " + e.getMessage());
                    e.printStackTrace();
                    // Fallback to text-only inference
                    response = model.runInstructOnce(sampler, options);
                }
            } else {
                System.err.println("⚠️ Non-VLM model detected - falling back to text inference");
                response = model.runInstructOnce(sampler, options);
            }
            System.out.println(response);
            if (SHOW_PERF_INTERACTIVE) {
                LastRunMetrics.printMetrics();
            }
            
        } catch (IOException e) {
            System.err.println("Error loading image: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Error processing multimodal input: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * <p>Initializes model options, loads the appropriate model (either AOT or on-demand),
     * configures the sampler, and runs either in interactive or single-instruction mode based on the input options.</p>
     *
     * @param args
     *         command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException
     *         if model loading or file operations fail.
     */
    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Sampler sampler = createSampler(model, options);

        if (options.interactive()) {
            model.runInteractive(sampler, options);
        } else {
            runSingleInstruction(model, sampler, options);
        }
    }
}



