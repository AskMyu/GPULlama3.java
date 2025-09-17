package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * Represents the state of the Granite model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Granite model.
 *
 * <p>Granite models have unique characteristics:</p>
 * <ul>
 *   <li>Grouped Query Attention (GQA) for efficient memory usage</li>
 *   <li>SwiGLU activation requiring additional intermediate buffers</li>
 *   <li>Large context window (128K tokens)</li>
 *   <li>Fill-in-the-Middle (FIM) support</li>
 * </ul>
 */
public final class GraniteState extends State {

    public GraniteState(Configuration config, int batchsize) {
        super(config, batchsize);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();

        // Allocation with Granite dimensions
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.dim());
        fields.v = ArrayFloatTensor.allocate(config.dim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with GQA (Grouped Query Attention)
        // KV dimension is reduced due to grouped attention
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                .limit(config.numberOfLayers())
                .toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                .limit(config.numberOfLayers())
                .toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Granite dimensions
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(config.dim());
        fields.wrapXb2 = new FloatArray(config.dim());
        
        // For SwiGLU activation, we need larger intermediate buffers
        // SwiGLU: gate_proj(x) * silu(up_proj(x)) 
        fields.wrapHb = new FloatArray(config.hiddenDim());  // Gate projection
        fields.wrapHb2 = new FloatArray(config.hiddenDim()); // Up projection

        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(config.dim());
        fields.wrapK = new FloatArray(config.dim());
        fields.wrapV = new FloatArray(config.dim());

        // GQA-aware key/value cache wrappers
        fields.wrapKeyCache = new FloatArray(config.contextLength() * kvDim * config.numberOfLayers());
        fields.wrapValueCache = new FloatArray(config.contextLength() * kvDim * config.numberOfLayers());
        // NOTE: Avoiding .init(0.f) calls to prevent TornadoVM memory bounds violations
        
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // Temporary arrays for reductions
        fields.temp = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));

        return fields;
    }
}