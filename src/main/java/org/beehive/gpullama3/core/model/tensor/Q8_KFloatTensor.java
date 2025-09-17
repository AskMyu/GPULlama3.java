package org.beehive.gpullama3.core.model.tensor;

import org.beehive.gpullama3.LlamaApp;
import org.beehive.gpullama3.core.model.GGMLType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q8_K} format.
 * <p>
 * Q8_K structure (QK_K=256): d[4] + qs[256] + bsums[32] = 292 bytes per block
 * - d: float scale factor (4 bytes)
 * - qs: int8_t quantized values (256 bytes)
 * - bsums: int16_t block sums for groups of 16 (16 * 2 = 32 bytes)
 */
public final class Q8_KFloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q8_KFloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q8_K;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q8_K.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_K.getTypeSize();
        int withinBlockIndex = index % GGMLType.Q8_K.getBlockSize();
        
        // Q8_K layout: d[4] + qs[256] + bsums[32] = 292 bytes per block
        
        // Read scale factor (d) at the beginning
        float d = readFloat(memorySegment, blockOffset);
        
        // Read quantized value from qs array (after d)
        int qsOffset = blockOffset + 4; // After d
        byte quant = readByte(memorySegment, qsOffset + withinBlockIndex);
        
        // Q8_K dequantization: x = d * quant
        // The bsums are used for optimization in SIMD implementations but not needed for basic dequantization
        return d * quant;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (LlamaApp.USE_VECTOR_API && that instanceof ArrayFloatTensor) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q8_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize()
        assert Integer.bitCount(GGMLType.Q8_K.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_K.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_K.getBlockSize() == 0;

        // Use scalar implementation for now - could be optimized later with vectorization
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}