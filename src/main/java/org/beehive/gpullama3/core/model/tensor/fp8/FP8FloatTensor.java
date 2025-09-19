package org.beehive.gpullama3.core.model.tensor.fp8;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * FP8 quantized float tensor implementation for DeepSeek-R1.
 *
 * FP8 (8-bit floating point) provides 2x FLOPS improvement over FP16
 * while maintaining reasonable precision for large model inference.
 *
 * ISOLATION: This is completely separate from existing quantization schemes.
 */
public class FP8FloatTensor extends FloatTensor {

    private final int[] shape;
    private final byte[] fp8Data;    // Raw FP8 data
    private final float scale;       // Scaling factor for quantization
    private final float zero_point;  // Zero point for asymmetric quantization
    private final long size;

    // FP8 format parameters (E4M3 format: 4 exponent bits, 3 mantissa bits)
    private static final int FP8_BIAS = 7;        // Exponent bias
    private static final int FP8_MAX_EXP = 15;    // Maximum exponent value
    private static final float FP8_MAX_VALUE = 448.0f;  // Maximum representable value
    private static final float FP8_MIN_NORMAL = 0.0078125f; // Minimum normal value

    public FP8FloatTensor(int[] shape, ByteBuffer buffer, float scale, float zero_point) {
        this.shape = shape.clone();
        this.scale = scale;
        this.zero_point = zero_point;

        // Calculate total elements
        long totalElements = 1;
        for (int dim : shape) {
            totalElements *= dim;
        }
        this.size = totalElements;

        // Extract FP8 data from buffer
        this.fp8Data = new byte[(int) size];
        buffer.get(fp8Data);
    }

    /**
     * Create FP8 tensor from float array (for testing/conversion).
     */
    public static FP8FloatTensor fromFloatArray(float[] data, int[] shape) {
        // Compute quantization parameters
        float min_val = Float.MAX_VALUE;
        float max_val = Float.MIN_VALUE;

        for (float value : data) {
            if (!Float.isNaN(value) && !Float.isInfinite(value)) {
                min_val = Math.min(min_val, value);
                max_val = Math.max(max_val, value);
            }
        }

        // Symmetric quantization for simplicity
        float scale = Math.max(Math.abs(min_val), Math.abs(max_val)) / FP8_MAX_VALUE;
        float zero_point = 0.0f; // Symmetric quantization

        // Quantize data
        byte[] fp8Data = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            fp8Data[i] = floatToFP8(data[i] / scale);
        }

        // Create buffer
        ByteBuffer buffer = ByteBuffer.allocate(data.length);
        buffer.put(fp8Data);
        buffer.flip();

        return new FP8FloatTensor(shape, buffer, scale, zero_point);
    }

    @Override
    public float getFloat(int index) {
        if (index >= this.size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + this.size);
        }

        byte fp8Value = fp8Data[index];
        float dequantized = fp8ToFloat(fp8Value);
        return dequantized * scale + zero_point;
    }

    @Override
    public void setFloat(int index, float value) {
        if (index >= this.size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + this.size);
        }

        float normalized = (value - zero_point) / scale;
        fp8Data[index] = floatToFP8(normalized);
    }

    @Override
    public int size() {
        return (int) this.size;
    }

    @Override
    public MemorySegment asMemorySegment() {
        // For FP8, we need to create a memory segment from the byte array
        return MemorySegment.ofArray(fp8Data);
    }

    public int[] shape() {
        return shape.clone();
    }

    public GGMLType ggmlType() {
        return GGMLType.F32; // Report as F32 for compatibility
    }

    @Override
    protected GGMLType type() {
        return GGMLType.F32; // Report as F32 for compatibility
    }

    @Override
    protected FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
        // Convert FP8 data to float array for vector operations
        float[] floatData = new float[species.length()];
        for (int i = 0; i < species.length() && (offset + i) < size; i++) {
            floatData[i] = getFloat(offset + i);
        }
        return FloatVector.fromArray(species, floatData, 0);
    }

    /**
     * Convert float to FP8 (E4M3 format).
     */
    private static byte floatToFP8(float value) {
        if (Float.isNaN(value)) {
            return 0x7F; // NaN representation
        }

        if (Float.isInfinite(value)) {
            return value > 0 ? 0x7F : (byte) 0xFF; // +/-Inf
        }

        if (value == 0.0f) {
            return 0x00;
        }

        // Handle sign
        boolean negative = value < 0;
        value = Math.abs(value);

        // Clamp to FP8 range
        if (value >= FP8_MAX_VALUE) {
            return (byte) (negative ? 0xFF : 0x7F);
        }

        if (value < FP8_MIN_NORMAL) {
            // Subnormal numbers (not fully implemented for simplicity)
            return 0x00;
        }

        // Extract exponent and mantissa
        int floatBits = Float.floatToIntBits(value);
        int exponent = ((floatBits >> 23) & 0xFF) - 127 + FP8_BIAS; // Adjust bias
        int mantissa = (floatBits >> 20) & 0x07; // Take top 3 bits of mantissa

        // Clamp exponent to FP8 range
        if (exponent <= 0) {
            // Underflow to zero
            return 0x00;
        }
        if (exponent > FP8_MAX_EXP) {
            // Overflow to max
            return (byte) (negative ? 0xFF : 0x7F);
        }

        // Assemble FP8 value: [sign][4-bit exp][3-bit mantissa]
        int fp8Value = (mantissa) | (exponent << 3);
        if (negative) {
            fp8Value |= 0x80; // Set sign bit
        }

        return (byte) fp8Value;
    }

    /**
     * Convert FP8 (E4M3 format) to float.
     */
    private static float fp8ToFloat(byte fp8Value) {
        int value = fp8Value & 0xFF;

        // Extract components
        boolean negative = (value & 0x80) != 0;
        int exponent = (value >> 3) & 0x0F;
        int mantissa = value & 0x07;

        // Handle special cases
        if (exponent == 0) {
            if (mantissa == 0) {
                return negative ? -0.0f : 0.0f;
            } else {
                // Subnormal (not fully implemented)
                return 0.0f;
            }
        }

        if (exponent == FP8_MAX_EXP) {
            // NaN or Infinity
            return negative ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
        }

        // Normal number
        // Convert to float: mantissa becomes 1.xxx * 2^(exp - bias)
        float mantissaFloat = 1.0f + (float) mantissa / 8.0f; // mantissa / 2^3
        float exponentFloat = (float) Math.pow(2.0, exponent - FP8_BIAS);

        float result = mantissaFloat * exponentFloat;
        return negative ? -result : result;
    }

    /**
     * Get raw FP8 data for GPU operations.
     */
    public byte[] getFP8Data() {
        return fp8Data.clone();
    }

    /**
     * Get quantization scale.
     */
    public float getScale() {
        return scale;
    }

    /**
     * Get zero point.
     */
    public float getZeroPoint() {
        return zero_point;
    }

    /**
     * Convert entire tensor to float array (for debugging/validation).
     */
    public float[] toFloatArray() {
        float[] result = new float[(int) size];
        for (int i = 0; i < size; i++) {
            result[i] = getFloat(i);
        }
        return result;
    }

    /**
     * Compute quantization error compared to original float array.
     */
    public float computeQuantizationError(float[] original) {
        if (original.length != size) {
            throw new IllegalArgumentException("Array size mismatch");
        }

        float totalError = 0.0f;
        for (int i = 0; i < size; i++) {
            float error = Math.abs(original[i] - getFloat(i));
            totalError += error;
        }

        return totalError / size; // Mean absolute error
    }

    /**
     * Get memory usage in bytes.
     */
    public long getMemoryUsage() {
        return size; // 1 byte per element + overhead
    }

    /**
     * Get compression ratio compared to FP32.
     */
    public float getCompressionRatio() {
        return 4.0f; // FP8 is 4x smaller than FP32
    }

    @Override
    public String toString() {
        return String.format("FP8FloatTensor{shape=%s, size=%d, scale=%.6f, zero_point=%.6f}",
            java.util.Arrays.toString(shape), size, scale, zero_point);
    }
}