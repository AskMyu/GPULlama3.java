package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.model.Model;

import java.nio.channels.FileChannel;

/**
 * Placeholder model loader for GPT-OSS models.
 * This is a stub implementation for future development.
 * Full MoE implementation requires significant additional infrastructure.
 */
public class GptOssModelLoader extends ModelLoader {

    public GptOssModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    public Model loadModel() {
        throw new UnsupportedOperationException(
            "GPT-OSS MoE implementation is not yet complete. " +
            "This requires complex Mixture-of-Experts infrastructure " +
            "and significant GPU memory (16GB+ recommended)."
        );
    }
}