package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.aot.AOT;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.F16FloatTensor;
import org.beehive.gpullama3.core.model.tensor.F32FloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.model.tensor.Q3_KFloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q4_0FloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q4_KFloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q5_KFloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q6_KFloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q8_KFloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q8_0FloatTensor;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.LlamaStandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.IntFunction;

public abstract class ModelLoader {

    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));               // Use Ahead-of-Time compilation

    protected FileChannel fileChannel;
    protected GGUF gguf;
    protected int contextLength;
    protected boolean loadWeights;
    protected boolean useTornadovm;
    protected String modelPath;

    public ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.loadWeights = loadWeights;
        this.useTornadovm = useTornadovm;
        this.modelPath = null;
    }

    public ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, String modelPath) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.loadWeights = loadWeights;
        this.useTornadovm = true; // Default to TornadoVM for VLM models
        this.modelPath = modelPath;
    }

    public ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm, String modelPath) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.loadWeights = loadWeights;
        this.useTornadovm = useTornadovm;
        this.modelPath = modelPath;
    }

    private static ModelType detectModelType(Map<String, Object> metadata, Path ggufPath) {
        String name = (String) metadata.get("general.name");
        String architecture = (String) metadata.get("general.architecture");
        String tokenizerModel = (String) metadata.get("tokenizer.ggml.model");
        Integer vocabSize = (Integer) metadata.get("llama.vocab_size");
        String filename = ggufPath.getFileName().toString().toLowerCase();
        String fullPath = ggufPath.toString().toLowerCase();


        // Check by architecture first (more reliable)
        if (architecture != null) {
            String lowerArch = architecture.toLowerCase();
            System.err.printf("[MODEL-LOADER] Architecture detected: '%s'%n", architecture);
            if (lowerArch.equals("gemma") || lowerArch.equals("gemma2") || lowerArch.equals("gemma3")) {
                // GEMMA 2B SUPPORT REMOVED - all Gemma models now use GEMMA_3 loader
                System.err.printf("[MODEL-LOADER] ✅ Detected GEMMA architecture: %s -> GEMMA_3 (Gemma 2B support removed)%n", architecture);
                return ModelType.GEMMA_3;
            } else if (lowerArch.equals("gptoss") || lowerArch.equals("gpt-oss")) {
                return ModelType.GPT_OSS;
            } else if (lowerArch.equals("granite")) {
                return ModelType.GRANITE_3_3;
            } else if (lowerArch.equals("phi4") || lowerArch.equals("phi-4")) {
                return ModelType.PHI_4_MINI_REASONING;
            } else if (lowerArch.equals("olmoe") || lowerArch.equals("olmo")) {
                return ModelType.OLMOE_1B_7B;
            }
        }

        // Check filename and path for LLaVA models (since they often don't identify as LLaVA in metadata)
        if ((filename != null && (filename.contains("llava") || filename.contains("llava-llama") || filename.contains("llava-v"))) ||
            (fullPath != null && (fullPath.contains("llava") || fullPath.contains("llava-phi") || fullPath.contains("llava-llama")))) {
            if (filename.contains("v1.5") || filename.contains("1.5") || filename.contains("v1_5")) {
                // LLaVA-1.5 variant - use Llama3-8B type for compatibility
                return ModelType.LLAVA_LLAMA_3_8B_INT4; // LLaVA-1.5-7B with Q4 quantization
            } else if (fullPath.contains("llava-phi") || filename.contains("phi")) {
                // LLaVA-Phi variant - use Llama3-8B type for compatibility since it's multimodal
                if (filename.contains("int4") || filename.contains("q4") || filename.contains("4_k")) {
                    return ModelType.LLAVA_LLAMA_3_8B_INT4;
                }
                return ModelType.LLAVA_LLAMA_3_8B;
            } else if (filename.contains("llama-3") || filename.contains("llama3")) {
                // LLaVA-Llama3 variant
                if (filename.contains("int4") || filename.contains("q4") || filename.contains("4_k")) {
                    return ModelType.LLAVA_LLAMA_3_8B_INT4;
                }
                return ModelType.LLAVA_LLAMA_3_8B;
            } else {
                // Generic LLaVA - assume Llama3 variant
                if (filename.contains("int4") || filename.contains("q4") || filename.contains("4_k")) {
                    return ModelType.LLAVA_LLAMA_3_8B_INT4;
                }
                return ModelType.LLAVA_LLAMA_3_8B;
            }
        }

        // Check by name as fallback
        if (name != null) {
            String lowerName = name.toLowerCase();
            System.err.printf("[MODEL-LOADER] Model name: '%s'%n", name);
            if (lowerName.contains("gemma")) {
                // GEMMA 2B SUPPORT REMOVED - all Gemma models now use GEMMA_3 loader
                System.err.printf("[MODEL-LOADER] ✅ Detected GEMMA by name: %s -> GEMMA_3 (Gemma 2B support removed)%n", name);
                return ModelType.GEMMA_3;
            } else if (lowerName.contains("gpt-oss") || lowerName.contains("gptoss")) {
                return ModelType.GPT_OSS;
            } else if (lowerName.contains("granite-3") || lowerName.contains("granite3") || lowerName.contains("granite")) {
                return ModelType.GRANITE_3_3;
            } else if (lowerName.contains("deepseek-r1-distill-qwen-1.5b") ||
                       lowerName.contains("deepseek r1 distill qwen 1.5b")) {
                return ModelType.DEEPSEEK_R1_DISTILL_QWEN_1_5B;
            } else if (lowerName.contains("deepseek-r1-distill-qwen-14b") ||
                       lowerName.contains("deepseek r1 distill qwen 14b")) {
                return ModelType.DEEPSEEK_R1_DISTILL_QWEN_14B;
            } else if (lowerName.contains("deepseek r1 distill") ||
                       lowerName.contains("deepseek-r1-distill")) {
                return ModelType.DEEPSEEK_R1_DISTILL_QWEN;
            } else if (lowerName.contains("olmoe-1b-7b") || 
                       lowerName.contains("olmoe 1b 7b")) {
                return ModelType.OLMOE_1B_7B;
            } else if (lowerName.contains("phi-4-mini-reasoning") || 
                       lowerName.contains("phi4-mini-reasoning") ||
                       lowerName.contains("phi-4 mini reasoning")) {
                return ModelType.PHI_3; // Use PHI_3 since architecture is phi3
            } else if (lowerName.contains("qwen3-30b-a3b") || 
                       lowerName.contains("qwen3 30b a3b")) {
                return ModelType.QWEN3_30B_A3B;
            } else if (lowerName.contains("mistral")) {
                return ModelType.MISTRAL;
            } else if (lowerName.contains("llava") || lowerName.contains("llava-llama") || lowerName.contains("llava-v")) {
                // Detect LLaVA model variant by name - MUST come before generic "llama" check
                if (lowerName.contains("v1.5") || lowerName.contains("1.5") || lowerName.contains("v1_5")) {
                    // LLaVA-1.5 variant - use Llama3-8B type for compatibility
                    return ModelType.LLAVA_LLAMA_3_8B_INT4; // LLaVA-1.5-7B with Q4 quantization
                } else if (lowerName.contains("llama-3") || lowerName.contains("llama3")) {
                    // LLaVA-Llama3 variant
                    if (lowerName.contains("int4") || lowerName.contains("q4") || lowerName.contains("4_k")) {
                        return ModelType.LLAVA_LLAMA_3_8B_INT4;
                    }
                    return ModelType.LLAVA_LLAMA_3_8B;
                } else {
                    // Generic LLaVA - assume Llama3 variant
                    if (lowerName.contains("int4") || lowerName.contains("q4") || lowerName.contains("4_k")) {
                        return ModelType.LLAVA_LLAMA_3_8B_INT4;
                    }
                    return ModelType.LLAVA_LLAMA_3_8B;
                }
            } else if (lowerName.contains("llama")) {
                return ModelType.LLAMA_3;
            } else if (lowerName.contains("qwen2")) {
                return ModelType.QWEN_2;
            } else if (lowerName.contains("qwen3")) {
                return ModelType.QWEN_3;
            } else if (lowerName.contains("phi3")) {
                return ModelType.PHI_3;
            } else if (lowerName.contains("llava") || lowerName.contains("llava-llama") || lowerName.contains("llava-v")) {
                // Detect LLaVA model variant
                if (lowerName.contains("v1.5") || lowerName.contains("1.5") || lowerName.contains("v1_5")) {
                    // LLaVA-1.5 variant - use Llama3-8B type for now (same architecture)
                    return ModelType.LLAVA_LLAMA_3_8B_INT4; // LLaVA-1.5-7B with Q4 quantization
                } else if (lowerName.contains("llama-3") || lowerName.contains("llama3")) {
                    // LLaVA-Llama3 variant
                    if (lowerName.contains("int4") || lowerName.contains("q4") || lowerName.contains("4_k")) {
                        return ModelType.LLAVA_LLAMA_3_8B_INT4;
                    }
                    return ModelType.LLAVA_LLAMA_3_8B;
                } else {
                    // Generic LLaVA - assume Llama3 variant
                    if (lowerName.contains("int4") || lowerName.contains("q4") || lowerName.contains("4_k")) {
                        return ModelType.LLAVA_LLAMA_3_8B_INT4;
                    }
                    return ModelType.LLAVA_LLAMA_3_8B;
                }
            } else if (lowerName.contains("smolvlm") || lowerName.contains("smol-vlm") || 
                       lowerName.contains("smolvlm2") || lowerName.contains("smolvlm-500m")) {
                // SmolVLM model - check if it exists in ModelType
                // return ModelType.SMOL_VLM_500M;
            }
        }
        
        // Check filename if metadata doesn't provide clear identification
        if (filename != null) {
            if (filename.contains("llava") || filename.contains("llava-llama") || filename.contains("llava-v")) {
                // Detect LLaVA model variant by filename
                if (filename.contains("v1.5") || filename.contains("1.5") || filename.contains("v1_5")) {
                    // LLaVA-1.5 variant - use Llama3-8B type for compatibility
                    return ModelType.LLAVA_LLAMA_3_8B_INT4; // LLaVA-1.5-7B with Q4 quantization
                } else if (filename.contains("llama-3") || filename.contains("llama3")) {
                    // LLaVA-Llama3 variant
                    if (filename.contains("int4") || filename.contains("q4") || filename.contains("4_k")) {
                        return ModelType.LLAVA_LLAMA_3_8B_INT4;
                    }
                    return ModelType.LLAVA_LLAMA_3_8B;
                } else {
                    // Generic LLaVA - assume Llama3 variant
                    if (filename.contains("int4") || filename.contains("q4") || filename.contains("4_k")) {
                        return ModelType.LLAVA_LLAMA_3_8B_INT4;
                    }
                    return ModelType.LLAVA_LLAMA_3_8B;
                }
            }
        }

        return ModelType.UNKNOWN;
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
    public static Model loadModel(Options options) throws IOException {
        if (USE_AOT) {
            Model model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
            if (model == null) {
                throw new IllegalStateException("Failed to load precompiled AOT model.");
            }
            return model;
        }
        return ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
    }

    public static Model loadModel(Path ggufPath, int contextLength, boolean loadWeights, boolean useTornadovm) throws IOException {
        // initial load of metadata from gguf file
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        // detect model type
        ModelType modelType = detectModelType(gguf.getMetadata(), ggufPath);

        // model type-specific load - pass model path for VLM models
        if (modelType.isVisionLanguageModel()) {
            return modelType.loadModel(fileChannel, gguf, contextLength, loadWeights, useTornadovm, ggufPath.toString());
        } else {
            return modelType.loadModel(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
        }
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();

        // CRITICAL DEBUG: Extensive debugging for OLMoE key tensors
        if (entry.name().contains("output.weight") || entry.name().contains("wcls") || entry.name().contains("token_embd")) {
            System.err.printf("[LOAD-QUANTIZED-DEBUG] Loading tensor '%s' with ggmlType=%s (ordinal=%d)%n",
                            entry.name(), ggmlType, ggmlType.ordinal());
            System.err.printf("[LOAD-QUANTIZED-DEBUG] Shape: %s, numElements: %d, memorySegment size: %d%n",
                            java.util.Arrays.toString(entry.shape()),
                            FloatTensor.numberOfElements(entry.shape()),
                            entry.memorySegment().byteSize());

            // Check memory segment address and offset information
            var memSeg = entry.memorySegment();
            System.err.printf("[LOAD-QUANTIZED-DEBUG] MemorySegment address: %s, native address: %s%n",
                            memSeg.address(),
                            memSeg.isNative() ? "native" : "heap");

            // Compare with mapped file info
            var mappedFile = entry.mappedFile();
            System.err.printf("[LOAD-QUANTIZED-DEBUG] MappedFile size: %d, tensor offset: %d%n",
                            mappedFile.byteSize(),
                            memSeg.address() - mappedFile.address());

            // Check raw memory segment data
            if (memSeg.byteSize() >= 16) {
                System.err.printf("[LOAD-QUANTIZED-DEBUG] Raw bytes (first 16): %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x%n",
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 0),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 1),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 2),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 3),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 4),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 5),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 6),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 7),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 8),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 9),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 10),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 11),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 12),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 13),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 14),
                    memSeg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, 15));
            }
        }
        
        return switch (ggmlType) {
            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q3_K -> new Q3_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_K -> new Q4_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q5_K -> new Q5_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q6_K -> new Q6_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_K -> new Q8_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case IQ3_T -> {
                // IQ3_T: Native ternary quantization support
                System.err.printf("[IQ3_T-NATIVE] Loading IQ3_T tensor '%s' with ternary decoding%n", entry.name());
                yield new org.beehive.gpullama3.core.model.tensor.IQ3_TFloatTensor(
                    FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            }
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatArray[] loadArrayAsFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }
    //@formatter:on

    public static HalfFloatArray[] loadArrayAsHalfFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        HalfFloatArray[] array = new HalfFloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsHalfFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    //@formatter:off

    public static FloatArray floatBufferToFloatArray(GGMLTensorEntry tensorEntry) {
        if (tensorEntry.ggmlType() == GGMLType.F32) {
            FloatBuffer buffer = tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            return FloatArray.fromFloatBuffer(buffer);
        } else {
            throw new UnsupportedOperationException("Conversion to FloatArray from " + tensorEntry.ggmlType());
        }
    }
    //@formatter:on

    public static FloatArray[] loadArrayAsFloatArrayFromBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            if (entry == null) {
                System.err.println("ERROR: Tensor entry is null at index " + i);
                throw new RuntimeException("Missing tensor at index " + i);
            }
            array[i] = floatBufferToFloatArray(entry);
        }
        return array;
    }

    public static ByteArray createByteArrayFromTensor(GGMLTensorEntry entry) {
        FloatTensor tensor = loadQuantized(entry);
        return ByteArray.fromSegment(tensor.asMemorySegment());
    }

    public static FloatArray loadTensorAsFloatArray(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.F32) {
            // For F32, we can directly create FloatArray from memory
            FloatBuffer buffer = entry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            FloatArray array = new FloatArray(buffer.remaining());
            for (int i = 0; i < buffer.remaining(); i++) {
                array.set(i, buffer.get());
            }
            return array;
        } else {
            // For quantized formats, use multi-threaded dequantization
            FloatTensor tensor = loadQuantized(entry);
            return loadTensorAsFloatArrayMultiThreaded(tensor, entry.name());
        }
    }

    /**
     * Multi-threaded tensor dequantization with configurable thread count.
     * Thread count logic:
     * - availableProcessors > 6: use (availableProcessors - 4)
     * - availableProcessors <= 6: cap at 2
     */
    private static FloatArray loadTensorAsFloatArrayMultiThreaded(FloatTensor tensor, String tensorName) {
        int tensorSize = tensor.size();

        // CRITICAL FIX: Check for integer overflow causing negative allocation sizes
        if (tensorSize < 0) {
            System.err.printf("[ALLOCATION-ERROR] Tensor '%s' has negative size %d (integer overflow)%n", tensorName, tensorSize);
            System.err.printf("[ALLOCATION-ERROR] This indicates the tensor is too large for current int-based allocation%n");
            throw new IllegalArgumentException("Tensor too large for allocation: " + tensorName + " (size overflow: " + tensorSize + ")");
        }

        // 🚀 FLOATARRAY-LONGFIX: TornadoVM limitation has been resolved
        // Previous block that prevented >2GB allocations has been removed
        // TornadoVM FloatArray now supports large tensor allocations via integer overflow fix

        // Note: Artificial tensor size limit removed after FloatArrayLongFix implementation
        // The TornadoVM framework now properly handles tensors >2GB with long arithmetic
        // Previous limit of 650_000_000L (~2.6GB) is no longer needed

        System.err.printf("[TENSOR-LOAD] Loading tensor '%s' with size %d%n", tensorName, tensorSize);
        FloatArray array = new FloatArray(tensorSize);

        // Get configured thread count
        int threadCount = getOptimalThreadCount();

        // For small tensors, use single-threaded to avoid overhead
        if (tensorSize < 1000 || threadCount <= 1) {
            System.err.printf("[Q4K-LOAD] Single-threaded: %s (%d elements)%n", tensorName, tensorSize);
            for (int i = 0; i < tensorSize; i++) {
                array.set(i, tensor.getFloat(i));
            }
            return array;
        }

        System.err.printf("[Q4K-LOAD] Multi-threaded (%d threads): %s (%d elements)%n",
                         threadCount, tensorName, tensorSize);

        // Create thread pool for parallel processing
        java.util.concurrent.ForkJoinPool customThreadPool = new java.util.concurrent.ForkJoinPool(threadCount);
        try {
            // Split work into chunks for parallel processing
            int chunkSize = Math.max(256, tensorSize / threadCount); // Minimum 256 elements per chunk for efficiency
            java.util.List<java.util.concurrent.CompletableFuture<Void>> futures = new java.util.ArrayList<>();

            for (int start = 0; start < tensorSize; start += chunkSize) {
                final int chunkStart = start;
                final int chunkEnd = Math.min(start + chunkSize, tensorSize);

                java.util.concurrent.CompletableFuture<Void> future = java.util.concurrent.CompletableFuture.runAsync(() -> {
                    for (int i = chunkStart; i < chunkEnd; i++) {
                        array.set(i, tensor.getFloat(i));
                    }
                }, customThreadPool);

                futures.add(future);
            }

            // Wait for all chunks to complete
            java.util.concurrent.CompletableFuture.allOf(futures.toArray(new java.util.concurrent.CompletableFuture[0])).join();

        } finally {
            customThreadPool.shutdown();
        }

        return array;
    }

    /**
     * Get optimal thread count for Q4K dequantization based on system configuration.
     */
    private static int getOptimalThreadCount() {
        // Check system property first
        String threadProp = System.getProperty("q4k.dequant.threads");
        if (threadProp != null) {
            try {
                int threads = Integer.parseInt(threadProp);
                if (threads == 0) return 1; // Single-threaded
                if (threads > 0) return threads; // Explicit thread count
                // threads < 0 falls through to auto-detection
            } catch (NumberFormatException e) {
                System.err.println("[Q4K-THREADS] Invalid q4k.dequant.threads value: " + threadProp + ", using auto-detection");
            }
        }

        // Auto-detection logic
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int optimalThreads;

        if (availableProcessors <= 6) {
            optimalThreads = Math.min(2, availableProcessors); // Cap at 2 for low-core systems
        } else {
            optimalThreads = availableProcessors - 4; // Leave 4 cores for system
        }

        System.err.printf("[Q4K-THREADS] Auto-detected: %d available processors, using %d threads for dequantization%n",
                         availableProcessors, optimalThreads);

        return Math.max(1, optimalThreads); // Ensure at least 1 thread
    }

    public static HalfFloatArray loadTensorAsHalfFloatArray(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.F32) {
            System.out.println("Loading F32 tensor as HalfFloatArray");
            return null;
        } else {
            // For quantized formats, use multi-threaded dequantization
            FloatTensor tensor = loadQuantized(entry);
            return loadTensorAsHalfFloatArrayMultiThreaded(tensor, entry.name());
        }
    }

    /**
     * Multi-threaded tensor dequantization to HalfFloatArray.
     */
    private static HalfFloatArray loadTensorAsHalfFloatArrayMultiThreaded(FloatTensor tensor, String tensorName) {
        int tensorSize = tensor.size();
        HalfFloatArray array = new HalfFloatArray(tensorSize);

        // Get configured thread count
        int threadCount = getOptimalThreadCount();

        // For small tensors, use single-threaded to avoid overhead
        if (tensorSize < 1000 || threadCount <= 1) {
            System.err.printf("[Q4K-LOAD-HALF] Single-threaded: %s (%d elements)%n", tensorName, tensorSize);
            for (int i = 0; i < tensorSize; i++) {
                HalfFloat x = new HalfFloat(tensor.getFloat(i));
                array.set(i, x);
            }
            return array;
        }

        System.err.printf("[Q4K-LOAD-HALF] Multi-threaded (%d threads): %s (%d elements)%n",
                         threadCount, tensorName, tensorSize);

        // Create thread pool for parallel processing
        java.util.concurrent.ForkJoinPool customThreadPool = new java.util.concurrent.ForkJoinPool(threadCount);
        try {
            // Split work into chunks for parallel processing
            int chunkSize = Math.max(256, tensorSize / threadCount);
            java.util.List<java.util.concurrent.CompletableFuture<Void>> futures = new java.util.ArrayList<>();

            for (int start = 0; start < tensorSize; start += chunkSize) {
                final int chunkStart = start;
                final int chunkEnd = Math.min(start + chunkSize, tensorSize);

                java.util.concurrent.CompletableFuture<Void> future = java.util.concurrent.CompletableFuture.runAsync(() -> {
                    for (int i = chunkStart; i < chunkEnd; i++) {
                        HalfFloat x = new HalfFloat(tensor.getFloat(i));
                        array.set(i, x);
                    }
                }, customThreadPool);

                futures.add(future);
            }

            // Wait for all chunks to complete
            java.util.concurrent.CompletableFuture.allOf(futures.toArray(new java.util.concurrent.CompletableFuture[0])).join();

        } finally {
            customThreadPool.shutdown();
        }

        return array;
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }

    public abstract Model loadModel();

    //@formatter:off
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        RopeConfig ropeConfig = new RopeConfig(8.0f,         // scaleFactor
                1.0f,                    // loFreqFactor
                3.0f,                    // hiFreqFactor
                8192                     // oldContextLength
        );

        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLength(),         // Maximum sequence length the model can process
                config.headSize(),              // Dimension of each attention head
                config.ropeTheta(),             // Base frequency parameter (typically 10000.0)
                ropeScaling,                    // Whether to apply frequency scaling (determined by model type)
                ropeConfig.scaleFactor,         // Scale factor for extending context length (NTK-aware scaling)
                ropeConfig.loFreqFactor,        // Low frequency scaling factor for better long-range dependencies
                ropeConfig.hiFreqFactor,        // High frequency scaling factor for preserving local precision
                ropeConfig.oldContextLength     // Original context length the model was trained with
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault("output.weight", tokenEmbeddings);

        System.err.printf("[MODELLOADER-DEBUG] useTornadovm=%b, will load %s weights%n",
                          useTornadovm, useTornadovm ? "TornadoVM" : "Standard");

        if (useTornadovm) {
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
            }
            System.err.println("[MODELLOADER-DEBUG] Creating TornadoVMWeights");
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            System.err.println("[MODELLOADER-DEBUG] Creating StandardWeights");
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new LlamaTornadoWeights(
                // Load directly to TornadoVM format
                loadTensorAsFloatArray(tokenEmbeddings), loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()), FloatArray.fromArray(ropeFreqs.second()), loadTensorAsHalfFloatArray(outputWeight), outputWeight.ggmlType()) {
        };
    }

    /**
     * Creates weights in standard format only
     */
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new LlamaStandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType());
    }

    // Helper class to encapsulate RoPE configuration parameters
    private static class RopeConfig {
        final float scaleFactor;
        final float loFreqFactor;
        final float hiFreqFactor;
        final int oldContextLength;

        RopeConfig(float scaleFactor, float loFreqFactor, float hiFreqFactor, int oldContextLength) {
            this.scaleFactor = scaleFactor;
            this.loFreqFactor = loFreqFactor;
            this.hiFreqFactor = hiFreqFactor;
            this.oldContextLength = oldContextLength;
        }
    }

}
