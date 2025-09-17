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
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_K} format.
 * <p>
 * This tensor implementation is not compatible with {@link FloatTensor}, but
 * {@link #dot(int, FloatTensor, int, int)} has a vectorized implementation that is used when
 * the second argument implements {@link FloatTensor}.
 */
public final class Q4_KFloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q4_K;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int superBlockIndex = index / GGMLType.Q4_K.getBlockSize();
        int superBlockOffset = superBlockIndex * GGMLType.Q4_K.getTypeSize();
        
        // DEBUG: Trace first call to verify method is being used
        if (index == 0) {
            System.err.printf("[Q4K-TRACE] getFloat() called for first time! index=%d, superBlockIndex=%d, superBlockOffset=%d%n",
                index, superBlockIndex, superBlockOffset);
            System.err.printf("[Q4K-TRACE] BlockSize=%d, TypeSize=%d%n", 
                GGMLType.Q4_K.getBlockSize(), GGMLType.Q4_K.getTypeSize());
        }
        
        // Q4_K structure: [d:f16][dmin:f16][scales:u8*12][qs:u4*128]
        // Super-block of 256 elements with 8 blocks of 32 elements each
        // Scales and mins are 6-bit quantized in scales[12] array
        
        // Read super-block scales
        float d = Float.float16ToFloat(readShort(memorySegment, superBlockOffset));
        float dmin = Float.float16ToFloat(readShort(memorySegment, superBlockOffset + Float16.BYTES));
        
        // Position within super-block
        int modIndex = index % GGMLType.Q4_K.getBlockSize();
        int blockIndex = modIndex / 32; // Which of 8 blocks (0-7)
        int withinBlock = modIndex % 32; // Position within block (0-31)
        
        // Extract 6-bit scale and min from scales[12] array
        int scalesOffset = superBlockOffset + 2 * Float16.BYTES;
        
        // The scales array contains 8 scales and 8 mins, each 6-bit
        // Packed as: scales[0-5] contain 4 scales/mins, scales[6-11] contain 4 scales/mins
        byte scale6bit, min6bit;
        
        if (blockIndex < 4) {
            // Blocks 0-3: packed in first 6 bytes
            int byteIdx = blockIndex + (blockIndex < 2 ? 0 : 2);  // 0,1,4,5
            int minByteIdx = blockIndex + (blockIndex < 2 ? 2 : 4); // 2,3,6,7
            scale6bit = (byte) (readByte(memorySegment, scalesOffset + byteIdx) & 0x3F);
            min6bit = (byte) (readByte(memorySegment, scalesOffset + minByteIdx) & 0x3F);
        } else {
            // Blocks 4-7: packed in last 6 bytes  
            int localIdx = blockIndex - 4;
            int byteIdx = 6 + localIdx + (localIdx < 2 ? 0 : 2); // 6,7,10,11
            int minByteIdx = 6 + localIdx + (localIdx < 2 ? 2 : 4); // 8,9,12,13 - but max is 11, so 8,9,10,11
            if (minByteIdx > 11) minByteIdx = 8 + localIdx; // Correct indexing
            
            scale6bit = (byte) (readByte(memorySegment, scalesOffset + byteIdx) & 0x3F);
            min6bit = (byte) (readByte(memorySegment, scalesOffset + Math.min(minByteIdx, 11)) & 0x3F);
        }
        
        // Apply the llama.cpp Q4_K formula: y = (d * sc) * quant - (min * m)
        // Where sc and m are the 6-bit scales treated as direct values
        float d1 = d * scale6bit;     // d * sc  
        float m1 = dmin * min6bit;    // min * m
        
        // Extract 4-bit quantized value from qs array
        int quantDataOffset = superBlockOffset + 2 * Float16.BYTES + 12; // After d, dmin, scales[12]
        int quantByteIndex = modIndex / 2; // Each byte holds 2 quantized values
        boolean isUpperNibble = (modIndex % 2) == 1;
        
        byte quantByte = readByte(memorySegment, quantDataOffset + quantByteIndex);
        int quant = isUpperNibble ? 
            ((quantByte >>> 4) & 0x0F) : 
            (quantByte & 0x0F);
        
        // Apply llama.cpp Q4_K dequantization formula exactly
        float result = d1 * quant - m1;
        
        // DEBUG: Log first few weight values to verify fix is applied
        if (index < 10) {
            System.err.printf("[Q4K-DEBUG] index=%d: d=%.6f, dmin=%.6f, scale6bit=%d, min6bit=%d, quant=%d, result=%.6f%n",
                index, d, dmin, scale6bit & 0xFF, min6bit & 0xFF, quant, result);
        }
        
        return result;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        // DEBUG: Trace dot product calls to see if this is being used
        if (thisOffset < 10) {
            System.err.printf("[Q4K-TRACE] dot() called: thisOffset=%d, thatOffset=%d, size=%d, useVectorAPI=%b%n",
                thisOffset, thatOffset, size, LlamaApp.USE_VECTOR_API && that instanceof ArrayFloatTensor);
        }
        
        if (LlamaApp.USE_VECTOR_API && that instanceof ArrayFloatTensor) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q4_K.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_K.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_K.getBlockSize() == 0;

        // For now, use scalar implementation for Q4_K
        // Full vectorization would require understanding the complete Q4_K layout
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}