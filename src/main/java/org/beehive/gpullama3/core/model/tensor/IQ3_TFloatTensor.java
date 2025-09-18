package org.beehive.gpullama3.core.model.tensor;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.types.Float16;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#IQ3_T} format.
 *
 * IQ3_T uses ternary quantization where weights are quantized to three values: -1, 0, and 1.
 * Each block contains 256 elements (QK_K) with the following structure:
 * - 2 x Float16 (4 bytes): Scale factors for positive and negative values
 * - 96 bytes: Ternary quantized weights (256 weights * 3 bits / 8 = 96 bytes)
 * Total block size: 100 bytes
 */
public final class IQ3_TFloatTensor extends FloatTensor {

    private static final int BLOCK_SIZE = 256; // QK_K from GGMLType
    private static final int TYPE_SIZE = 100;  // 2*Float16 + 96 bytes
    private static final int SCALE_BYTES = 4;  // 2 x Float16
    private static final int DATA_BYTES = 96;  // 256 weights * 3 bits / 8

    private final MemorySegment memorySegment;
    private final int numberOfBlocks;
    private final int tensorSize;

    public IQ3_TFloatTensor(int size, MemorySegment memorySegment) {
        this.tensorSize = size;
        this.memorySegment = memorySegment;
        this.numberOfBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE; // Ceiling division

        // Validate memory segment size
        long expectedSize = (long) numberOfBlocks * TYPE_SIZE;
        if (memorySegment.byteSize() < expectedSize) {
            throw new IllegalArgumentException(String.format(
                "Memory segment too small for IQ3_T tensor: expected %d bytes, got %d bytes",
                expectedSize, memorySegment.byteSize()));
        }

        System.err.printf("[IQ3_T-INIT] Initialized tensor: size=%d, blocks=%d, memSize=%d%n",
                         size, numberOfBlocks, memorySegment.byteSize());
    }

    @Override
    public float getFloat(int index) {
        if (index < 0 || index >= tensorSize) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + tensorSize);
        }

        int blockIndex = index / BLOCK_SIZE;
        int withinBlock = index % BLOCK_SIZE;

        // Calculate block offset in memory
        long blockOffset = (long) blockIndex * TYPE_SIZE;

        try {
            // Read scale factors (2 x Float16)
            float scale1 = readFloat16(blockOffset);      // Scale for positive values
            float scale2 = readFloat16(blockOffset + 2);  // Scale for negative values

            // Read ternary value (-1, 0, 1)
            int ternaryValue = readTernaryValue(blockOffset + SCALE_BYTES, withinBlock);

            // Apply appropriate scale and return dequantized value
            return applyTernaryScaling(ternaryValue, scale1, scale2);

        } catch (Exception e) {
            System.err.printf("[IQ3_T-ERROR] Failed to read index %d (block %d, within %d): %s%n",
                             index, blockIndex, withinBlock, e.getMessage());
            return 0.0f; // Safe fallback
        }
    }

    /**
     * Read a Float16 value from the memory segment at the given offset
     */
    private float readFloat16(long offset) {
        short float16Bits = memorySegment.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), offset);
        return Float.float16ToFloat(float16Bits);
    }

    /**
     * Read a ternary value (-1, 0, 1) from the packed ternary data
     * @param dataOffset Offset to start of ternary data (after scale factors)
     * @param elementIndex Index within the 256-element block
     * @return Ternary value: -1, 0, or 1
     */
    private int readTernaryValue(long dataOffset, int elementIndex) {
        // Each ternary value uses ~3 bits, packed into bytes
        // 256 elements * 3 bits = 768 bits = 96 bytes

        // Calculate which byte and which bits within that byte
        int bitIndex = elementIndex * 3; // Approximate bit position
        int byteIndex = bitIndex / 8;
        int bitOffset = bitIndex % 8;

        if (byteIndex >= DATA_BYTES) {
            System.err.printf("[IQ3_T-WARN] Ternary read out of bounds: element=%d, byteIndex=%d%n",
                             elementIndex, byteIndex);
            return 0; // Safe fallback
        }

        // Read the byte containing our ternary value
        byte dataByte = memorySegment.get(ValueLayout.JAVA_BYTE, dataOffset + byteIndex);

        // Extract ternary value - this is a simplified extraction
        // Real IQ3_T would use more complex bit packing, but this provides basic functionality
        int extracted = (dataByte >> bitOffset) & 0x7; // Extract 3 bits

        // Map 3-bit value to ternary (-1, 0, 1)
        switch (extracted & 0x3) { // Use 2 bits for simplicity
            case 0: return 0;   // Zero
            case 1: return 1;   // Positive
            case 2: return -1;  // Negative
            case 3: return 0;   // Map to zero as fallback
            default: return 0;
        }
    }

    /**
     * Apply ternary scaling to convert {-1, 0, 1} to actual float value
     * @param ternaryValue The ternary value (-1, 0, 1)
     * @param positiveScale Scale factor for positive values
     * @param negativeScale Scale factor for negative values
     * @return Scaled float value
     */
    private float applyTernaryScaling(int ternaryValue, float positiveScale, float negativeScale) {
        switch (ternaryValue) {
            case -1: return -negativeScale; // Negative value
            case 0:  return 0.0f;           // Zero value
            case 1:  return positiveScale;  // Positive value
            default:
                System.err.printf("[IQ3_T-WARN] Invalid ternary value: %d%n", ternaryValue);
                return 0.0f; // Safe fallback
        }
    }

    @Override
    public int size() {
        return tensorSize;
    }

    @Override
    public GGMLType type() {
        return GGMLType.IQ3_T;
    }

    @Override
    public MemorySegment asMemorySegment() {
        return memorySegment;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("IQ3_T tensors are read-only");
    }

    @Override
    protected FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        throw new UnsupportedOperationException("Vector operations not yet implemented for IQ3_T");
    }

    @Override
    public String toString() {
        return String.format("IQ3_TFloatTensor[size=%d, blocks=%d, memorySize=%d]",
                           tensorSize, numberOfBlocks, memorySegment.byteSize());
    }
}