package org.beehive.gpullama3.core.model.tensor;

import org.beehive.gpullama3.LlamaApp;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.types.Float16;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q6_K} format.
 * <p>
 * This tensor implementation is not compatible with {@link FloatTensor}, but
 * {@link #dot(int, FloatTensor, int, int)} has a vectorized implementation that is used when
 * the second argument implements {@link FloatTensor}.
 */
public final class Q6_KFloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    protected FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q6_K;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q6_K.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q6_K.getTypeSize();
        int withinBlockIndex = index % GGMLType.Q6_K.getBlockSize();
        
        // Q6_K layout (QK_K=256): ql[128] + qh[64] + scales[16] + d[2]
        // Total: 128 + 64 + 16 + 2 = 210 bytes
        
        // Read super-block scale (d) at the end
        int dOffset = blockOffset + 128 + 64 + 16; // After ql, qh, scales
        float d = Float.float16ToFloat(readShort(memorySegment, dOffset));
        
        // Read scale for this sub-block (16 elements per sub-block)
        int scaleIndex = withinBlockIndex / 16;
        int scalesOffset = blockOffset + 128 + 64; // After ql, qh
        byte scale = readByte(memorySegment, scalesOffset + scaleIndex);
        
        // Get lower 4 bits from ql array
        int qlOffset = blockOffset;
        int qlIndex = withinBlockIndex / 2;
        int qlByte = readByte(memorySegment, qlOffset + qlIndex) & 0xFF;
        int ql;
        if ((withinBlockIndex & 1) == 0) {
            ql = qlByte & 0x0F; // Lower 4 bits for even indices
        } else {
            ql = (qlByte >>> 4) & 0x0F; // Upper 4 bits for odd indices
        }
        
        // Get upper 2 bits from qh array
        int qhOffset = blockOffset + 128; // After ql
        int qhIndex = withinBlockIndex / 4;
        int qhByte = readByte(memorySegment, qhOffset + qhIndex) & 0xFF;
        int qhShift = (withinBlockIndex & 3) * 2;
        int qh = (qhByte >>> qhShift) & 0x03;
        
        // Combine to get 6-bit quantized value
        int quant6 = ql | (qh << 4); // Combine lower 4 bits + upper 2 bits
        
        // Convert to signed and apply scaling
        // Q6_K: x = d * scale * (quant - 32)
        float signedQuant = quant6 - 32.0f; // Center around 0
        return d * scale * signedQuant;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (LlamaApp.USE_VECTOR_API && that instanceof ArrayFloatTensor) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q6_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q6_K.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q6_K.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q6_K.getBlockSize() == 0;

        // For now, use scalar implementation for Q6_K
        // Full vectorization would require understanding the complete Q6_K layout
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}