package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;
import uk.ac.manchester.tornado.api.math.TornadoMath;

/**
 * Helper class for FloatArray operations that mirror the functionality of FloatTensor methods.
 * Provides utility methods for common tensor operations on TornadoVM's FloatArray type.
 */
public final class FloatArrayUtils {

    private FloatArrayUtils() {
        // Utility class, not meant to be instantiated
    }

    /**
     * Divides all elements in the specified range of a FloatArray by a value in-place.
     * Mirrors the functionality of FloatTensor.divideInPlace().
     *
     * @param array The FloatArray to modify
     * @param start The starting index (inclusive)
     * @param end The ending index (exclusive)
     * @param value The value to divide by
     * @return The modified FloatArray for method chaining
     */
    public static FloatArray divideInPlace(FloatArray array, int start, int end, float value) {
        for (int i = start; i < end; i++) {
            array.set(i, array.get(i) / value);
        }
        return array;
    }

    /**
     * Divides all elements in a FloatArray by a value in-place.
     *
     * @param array The FloatArray to modify
     * @param value The value to divide by
     * @return The modified FloatArray for method chaining
     */
    public static FloatArray divideInPlace(FloatArray array, float value) {
        return divideInPlace(array, 0, array.getSize(), value);
    }

    /**
     * Applies the softmax function to a range of elements in a FloatArray in-place.
     * Mirrors the functionality of FloatTensor.softmaxInPlace().
     *
     * @param array The FloatArray to modify
     * @param start The starting index (inclusive)
     * @param end The ending index (exclusive)
     * @return The modified FloatArray for method chaining
     */
    public static FloatArray softmaxInPlace(FloatArray array, int start, int end) {
        // Find max value for numerical stability
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = start; i < end; i++) {
            float val = array.get(i);
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Apply exp(x-max) to each element and calculate sum
        double sum = 0.0; // Use double precision for sum accumulation
        for (int i = start; i < end; i++) {
            float logit = array.get(i) - maxVal; // Subtract max for stability
            float exp;
            if (TornadoVMSupport.isTornadoVMEnabled()) {
                // Use TornadoMath for GPU execution if possible
                exp = TornadoMath.exp(logit);
            } else {
                // Fallback to standard Math
                exp = (float) Math.exp(logit);
            }

            // Handle underflow: if exp is too small, set to minimum representable value
            if (exp == 0.0f && logit > -87.0f) { // -87 is approximate limit for float exp
                exp = Float.MIN_VALUE;
            }

            array.set(i, exp);
            sum += exp;
        }

        // Enhanced debug output
        System.err.printf("[SOFTMAX-DEBUG] Processing range [%d:%d], maxVal=%.6f, sum=%.15e%n",
            start, end, maxVal, sum);

        // Enhanced edge case handling with debug info
        if (sum == 0.0 || sum < 1e-30) {
            System.err.printf("[SOFTMAX-DEBUG] Sum underflow detected: sum=%.15e, maxVal=%.6f, range=[%.6f, %.6f]%n",
                sum, maxVal, array.get(start), array.get(Math.min(start + 10, end - 1)));
            // Use uniform distribution as fallback
            float uniformProb = 1.0f / (end - start);
            for (int i = start; i < end; i++) {
                array.set(i, uniformProb);
            }
        } else {
            // Normal case, divide by sum (cast to float)
            float sumF = (float) sum;
            System.err.printf("[SOFTMAX-DEBUG] Normal softmax: sum=%.15e, sumF=%.6f%n", sum, sumF);
            for (int i = start; i < end; i++) {
                float prob = array.get(i) / sumF;
                // Ensure no zero probabilities for non-zero logits
                if (prob == 0.0f) {
                    prob = Float.MIN_VALUE;
                }
                array.set(i, prob);
            }
        }

        return array;
    }

    /**
     * Applies the softmax function to all elements in a FloatArray in-place.
     *
     * @param array The FloatArray to modify
     * @return The modified FloatArray for method chaining
     */
    public static FloatArray softmaxInPlace(FloatArray array) {
        return softmaxInPlace(array, 0, array.getSize());
    }

    /**
     * Finds the index of the maximum value in a FloatArray.
     * Mirrors the functionality of FloatTensor.argmax().
     *
     * @param array The FloatArray to search
     * @param start The starting index (inclusive)
     * @param end The ending index (exclusive)
     * @return The index of the maximum value
     */
    public static int argmax(FloatArray array, int start, int end) {
        float maxValue = Float.NEGATIVE_INFINITY;
        int maxIndex = start;

        for (int i = start; i < end; i++) {
            float value = array.get(i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /**
     * Finds the index of the maximum value in a FloatArray.
     *
     * @param array The FloatArray to search
     * @return The index of the maximum value
     */
    public static int argmax(FloatArray array) {
        return argmax(array, 0, array.getSize());
    }

    /**
     * Helper class to check if TornadoVM is enabled.
     * This allows us to decide whether to use TornadoMath or standard Math.
     */
    private static class TornadoVMSupport {
        private static final boolean TORNADO_VM_ENABLED;

        static {
            boolean enabled;
            try {
                // Try to access a TornadoVM-specific class
                Class.forName("uk.ac.manchester.tornado.api.math.TornadoMath");
                // Check for system property
                enabled = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
            } catch (ClassNotFoundException e) {
                enabled = false;
            }
            TORNADO_VM_ENABLED = enabled;
        }

        static boolean isTornadoVMEnabled() {
            return TORNADO_VM_ENABLED;
        }
    }
}