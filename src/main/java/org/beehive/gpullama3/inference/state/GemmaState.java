package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import org.beehive.gpullama3.tornadovm.SmartCacheArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * Represents the state of the Gemma model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Gemma model.
 *
 * <p>Gemma models have unique characteristics:</p>
 * <ul>
 *   <li>Large vocabulary size (256K tokens)</li>
 *   <li>Unusual parameter distribution (170M embeddings, 100M transformer)</li>
 *   <li>Optimized for compact deployment</li>
 * </ul>
 */
public final class GemmaState extends State {

    public GemmaState(Configuration config, int batchsize) {
        super(config, batchsize);
        // Set appropriate localSize for Gemma models - using a conservative value for stability
        this.localSize = 64;
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();

        // Allocation with Gemma dimensions
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.dim());
        fields.v = ArrayFloatTensor.allocate(config.dim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        
        // Large vocabulary allocation for Gemma (256K tokens)
        // This is much larger than typical models, so we ensure proper allocation
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Gemma dimensions
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                .limit(config.numberOfLayers())
                .toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                .limit(config.numberOfLayers())
                .toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Gemma dimensions
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(config.dim());
        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());

        // Large logits array for Gemma's vocabulary
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(config.dim());
        fields.wrapK = new FloatArray(config.dim());
        fields.wrapV = new FloatArray(config.dim());

        // Key/value cache wrappers - use SmartCacheArray to handle >2GB allocations
        int cacheSize = config.contextLength() * kvDim * config.numberOfLayers();
        System.out.printf("[GEMMA-STATE] Creating key/value caches: %d floats (%.2f GB each)%n",
                        cacheSize, (cacheSize * 4.0) / (1024 * 1024 * 1024));
        fields.wrapKeyCache = new SmartCacheArray(cacheSize);
        fields.wrapValueCache = new SmartCacheArray(cacheSize);
        // NOTE: Avoiding .init(0.f) calls to prevent TornadoVM memory bounds violations
        // Arrays are automatically initialized to 0.0f
        
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // Temporary arrays for reductions
        fields.temp = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));

        // Add Gemma 3 model detection flags - set last elements to special marker values
        // This allows GPU kernels to reliably detect Gemma models and apply optimizations
        int tempSize = fields.temp.getSize();
        if (tempSize > 1) {
            fields.temp.set(tempSize - 1, -999.0f); // Gemma detection marker
            fields.tempFFN.set(tempSize - 1, -999.0f);
            fields.tempLogits.set(tempSize - 1, -999.0f);
        }

        // Additional Gemma 3 optimization hints
        if (tempSize > 2) {
            // Mark as Gemma 3 with 5:1 attention pattern
            fields.temp.set(tempSize - 2, -888.0f); // Gemma 3 variant marker
            fields.tempFFN.set(tempSize - 2, -888.0f);
            fields.tempLogits.set(tempSize - 2, -888.0f);
        }

        return fields;
    }
}