package org.beehive.gpullama3.model;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.model.loader.GemmaModelLoader;
import org.beehive.gpullama3.model.loader.GptOssModelLoader;
import org.beehive.gpullama3.model.loader.GraniteModelLoader;
import org.beehive.gpullama3.model.loader.LlamaModelLoader;
import org.beehive.gpullama3.model.loader.MistralModelLoader;
import org.beehive.gpullama3.model.loader.Phi3ModelLoader;
import org.beehive.gpullama3.model.loader.Qwen2ModelLoader;
import org.beehive.gpullama3.model.loader.OlmoeModelLoader;
import org.beehive.gpullama3.model.loader.Phi4ModelLoader;
import org.beehive.gpullama3.model.loader.Qwen3ModelLoader;
import org.beehive.gpullama3.model.loader.LlavaModelLoader;

import java.nio.channels.FileChannel;

/**
 * Enumerates the different types of models supported by GPULlama3.java. This enum helps in categorizing and handling model-specific
 * logic based on the type of model being used.
 *
 * <p><b>Usage:</b> Use {@code ModelType} to specify or retrieve the type of
 * large language model (LLM), such as Llama or Qwen3. This ensures clean and structured handling of model behaviors and configurations by
 * dispatching calls to the appropriate model loader for each
 *  model type.</p>
 *
 * <p>Each enum value represents a distinct model type, which might be used for
 * conditional logic, initialization, or resource allocation within GPULlama3.java.</p>
 */
public enum ModelType {
    LLAMA_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new LlamaModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    MISTRAL {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new MistralModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    QWEN_2 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    QWEN_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Qwen3ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    DEEPSEEK_R1_DISTILL_QWEN {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    PHI_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Phi3ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    GEMMA_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new GemmaModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    GPT_OSS {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new GptOssModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    GRANITE_3_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new GraniteModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    DEEPSEEK_R1_DISTILL_QWEN_1_5B {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    OLMOE_1B_7B {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new OlmoeModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    PHI_4_MINI_REASONING {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Phi4ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    QWEN3_30B_A3B {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new Qwen3ModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },


    LLAVA_LLAMA_3_8B {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new LlavaModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    LLAVA_LLAMA_3_8B_INT4 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            return new LlavaModelLoader(fileChannel, gguf, contextLength, loadWeights, useTornadovm).loadModel();
        }
    },

    UNKNOWN {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
            throw new UnsupportedOperationException("Cannot load unknown model type");
        }
    };

    // Abstract method that each enum constant must implement
    public abstract Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm);

    public boolean isDeepSeekR1() {
        return this == DEEPSEEK_R1_DISTILL_QWEN || this == DEEPSEEK_R1_DISTILL_QWEN_1_5B;
    }
    
    public boolean isReasoningModel() {
        return this == DEEPSEEK_R1_DISTILL_QWEN || this == DEEPSEEK_R1_DISTILL_QWEN_1_5B || this == PHI_4_MINI_REASONING;
    }
    
    public boolean isMoEModel() {
        return this == GPT_OSS || this == OLMOE_1B_7B || this == QWEN3_30B_A3B;
    }
    
    public boolean isVisionLanguageModel() {
        return this == LLAVA_LLAMA_3_8B || this == LLAVA_LLAMA_3_8B_INT4;
    }
}
