package org.beehive.gpullama3.model.loader;

import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.Arrays;

import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.GGMLType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Testing suite for GPU tensor conversion correctness validation.
 * Validates that all optimization phases produce identical results to the reference implementation.
 */
public class GPUTensorConverterTest {
    private static final Logger logger = Logger.getLogger(GPUTensorConverterTest.class.getName());

    // Test configurations
    private static final int[] TEST_SIZES = {100, 1000, 10000, 100000, 1000000};
    private static final float EPSILON = 1e-6f; // Tolerance for floating-point comparison

    /**
     * Run all correctness tests
     */
    public static boolean runAllTests() {
        logger.info("[GPU-CONVERT-TEST] Starting comprehensive correctness validation...");

        boolean allPassed = true;

        try {
            // Test 1: Parallel vs Sequential CPU conversion
            allPassed &= testParallelConversionCorrectness();

            // Test 2: Bulk copy vs element-wise copy
            allPassed &= testBulkCopyCorrectness();

            // Test 3: GPU preprocessing correctness
            allPassed &= testGPUPreprocessingCorrectness();

            // Test 4: Streaming pipeline correctness
            allPassed &= testStreamingPipelineCorrectness();

            // Test 5: Buffer pooling correctness
            allPassed &= testBufferPoolingCorrectness();

            // Test 6: Unsafe bulk copy correctness
            allPassed &= testUnsafeBulkCopyCorrectness();

            // Test 7: Edge cases
            allPassed &= testEdgeCases();

            // Test 8: Performance degradation detection
            allPassed &= testPerformanceBounds();

        } catch (Exception e) {
            logger.severe("[GPU-CONVERT-TEST] Test suite failed with exception: " + e.getMessage());
            e.printStackTrace();
            allPassed = false;
        }

        if (allPassed) {
            logger.info("[GPU-CONVERT-TEST] ✅ All correctness tests PASSED");
        } else {
            logger.severe("[GPU-CONVERT-TEST] ❌ Some tests FAILED - optimizations may be incorrect");
        }

        return allPassed;
    }

    /**
     * Test 1: Parallel vs Sequential CPU conversion produces identical results
     */
    private static boolean testParallelConversionCorrectness() {
        logger.info("[GPU-CONVERT-TEST] Testing parallel CPU conversion correctness...");

        for (int size : TEST_SIZES) {
            // Create test tensor with random data
            MockFloatTensor tensor = createTestTensor(size);

            // Convert with parallel disabled
            System.setProperty("gpu.tensor.conversion.parallel.cpu", "false");
            HalfFloatArray sequentialResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // Convert with parallel enabled
            System.setProperty("gpu.tensor.conversion.parallel.cpu", "true");
            HalfFloatArray parallelResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // Compare results
            if (!compareHalfFloatArrays(sequentialResult, parallelResult, "parallel-cpu", size)) {
                return false;
            }
        }

        logger.info("[GPU-CONVERT-TEST] ✅ Parallel CPU conversion correctness validated");
        return true;
    }

    /**
     * Test 2: Bulk copy vs element-wise copy produces identical results
     */
    private static boolean testBulkCopyCorrectness() {
        logger.info("[GPU-CONVERT-TEST] Testing bulk copy correctness...");

        for (int size : TEST_SIZES) {
            MockFloatTensor tensor = createTestTensor(size);

            // Test with bulk copy disabled
            System.setProperty("gpu.tensor.conversion.bulk.copy", "false");
            HalfFloatArray elementResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // Test with bulk copy enabled
            System.setProperty("gpu.tensor.conversion.bulk.copy", "true");
            HalfFloatArray bulkResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            if (!compareHalfFloatArrays(elementResult, bulkResult, "bulk-copy", size)) {
                return false;
            }
        }

        logger.info("[GPU-CONVERT-TEST] ✅ Bulk copy correctness validated");
        return true;
    }

    /**
     * Test 3: GPU preprocessing produces expected transformations
     */
    private static boolean testGPUPreprocessingCorrectness() {
        logger.info("[GPU-CONVERT-TEST] Testing GPU preprocessing correctness...");

        // Test with known values for predictable preprocessing
        float[] testValues = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 100000.0f, -100000.0f};
        MockFloatTensor tensor = new MockFloatTensor(testValues);

        // Test scaling
        System.setProperty("gpu.tensor.conversion.preprocessing", "true");
        System.setProperty("gpu.tensor.conversion.scale", "2.0");
        System.setProperty("gpu.tensor.conversion.offset", "1.0");
        System.setProperty("gpu.tensor.conversion.normalize", "true");

        HalfFloatArray result = GPUTensorConverter.convertToHalfFloatArray(tensor);

        // Validate specific transformations
        // Original: -2.0 -> Scale*2 + 1 = -3.0 -> Normalize = -1.0 -> Clamp = -1.0
        if (Math.abs(result.get(0).getFloat32() - (-1.0f)) > EPSILON) {
            logger.severe(String.format("[GPU-CONVERT-TEST] Preprocessing failed: expected -1.0, got %.6f",
                                       result.get(0).getFloat32()));
            return false;
        }

        // Reset preprocessing properties
        System.setProperty("gpu.tensor.conversion.preprocessing", "false");
        System.setProperty("gpu.tensor.conversion.scale", "1.0");
        System.setProperty("gpu.tensor.conversion.offset", "0.0");
        System.setProperty("gpu.tensor.conversion.normalize", "false");

        logger.info("[GPU-CONVERT-TEST] ✅ GPU preprocessing correctness validated");
        return true;
    }

    /**
     * Test 4: Streaming pipeline produces identical results to standard processing
     */
    private static boolean testStreamingPipelineCorrectness() {
        logger.info("[GPU-CONVERT-TEST] Testing streaming pipeline correctness...");

        for (int size : new int[]{50000, 100000, 500000}) { // Large enough for streaming
            MockFloatTensor tensor = createTestTensor(size);

            // Standard processing
            System.setProperty("gpu.tensor.conversion.streaming", "false");
            HalfFloatArray standardResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // Streaming processing
            System.setProperty("gpu.tensor.conversion.streaming", "true");
            System.setProperty("gpu.tensor.conversion.streaming.buffers", "3");
            HalfFloatArray streamingResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            if (!compareHalfFloatArrays(standardResult, streamingResult, "streaming", size)) {
                return false;
            }
        }

        logger.info("[GPU-CONVERT-TEST] ✅ Streaming pipeline correctness validated");
        return true;
    }

    /**
     * Test 5: Buffer pooling doesn't affect results
     */
    private static boolean testBufferPoolingCorrectness() {
        logger.info("[GPU-CONVERT-TEST] Testing buffer pooling correctness...");

        for (int size : TEST_SIZES) {
            MockFloatTensor tensor = createTestTensor(size);

            // Without buffer pooling
            System.setProperty("gpu.tensor.conversion.pool.buffers", "false");
            HalfFloatArray nopoolResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // With buffer pooling
            System.setProperty("gpu.tensor.conversion.pool.buffers", "true");
            HalfFloatArray pooledResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // Run multiple times to test pool reuse
            HalfFloatArray pooledResult2 = GPUTensorConverter.convertToHalfFloatArray(tensor);

            if (!compareHalfFloatArrays(nopoolResult, pooledResult, "buffer-pool-1", size) ||
                !compareHalfFloatArrays(nopoolResult, pooledResult2, "buffer-pool-2", size)) {
                return false;
            }
        }

        logger.info("[GPU-CONVERT-TEST] ✅ Buffer pooling correctness validated");
        return true;
    }

    /**
     * Test 6: Unsafe bulk copy produces identical results to safe copy
     */
    private static boolean testUnsafeBulkCopyCorrectness() {
        logger.info("[GPU-CONVERT-TEST] Testing Unsafe bulk copy correctness...");

        for (int size : TEST_SIZES) {
            MockFloatTensor tensor = createTestTensor(size);

            // Safe copy
            System.setProperty("gpu.tensor.conversion.unsafe.bulk", "false");
            HalfFloatArray safeResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            // Unsafe copy (if available)
            System.setProperty("gpu.tensor.conversion.unsafe.bulk", "true");
            HalfFloatArray unsafeResult = GPUTensorConverter.convertToHalfFloatArray(tensor);

            if (!compareHalfFloatArrays(safeResult, unsafeResult, "unsafe-bulk", size)) {
                return false;
            }
        }

        logger.info("[GPU-CONVERT-TEST] ✅ Unsafe bulk copy correctness validated");
        return true;
    }

    /**
     * Test 7: Edge cases and boundary conditions
     */
    private static boolean testEdgeCases() {
        logger.info("[GPU-CONVERT-TEST] Testing edge cases...");

        // Test empty tensor
        MockFloatTensor emptyTensor = new MockFloatTensor(new float[0]);
        try {
            HalfFloatArray result = GPUTensorConverter.convertToHalfFloatArray(emptyTensor);
            if (result.getSize() != 0) {
                logger.severe("[GPU-CONVERT-TEST] Empty tensor test failed");
                return false;
            }
        } catch (Exception e) {
            logger.severe("[GPU-CONVERT-TEST] Empty tensor caused exception: " + e.getMessage());
            return false;
        }

        // Test single element
        MockFloatTensor singleTensor = new MockFloatTensor(new float[]{3.14159f});
        HalfFloatArray singleResult = GPUTensorConverter.convertToHalfFloatArray(singleTensor);
        if (Math.abs(singleResult.get(0).getFloat32() - 3.14159f) > 0.001f) { // HalfFloat precision
            logger.severe("[GPU-CONVERT-TEST] Single element test failed");
            return false;
        }

        // Test extreme values
        float[] extremeValues = {Float.MAX_VALUE, Float.MIN_VALUE, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        MockFloatTensor extremeTensor = new MockFloatTensor(extremeValues);
        try {
            HalfFloatArray extremeResult = GPUTensorConverter.convertToHalfFloatArray(extremeTensor);
            // Should not crash, values should be clamped appropriately
        } catch (Exception e) {
            logger.severe("[GPU-CONVERT-TEST] Extreme values caused exception: " + e.getMessage());
            return false;
        }

        logger.info("[GPU-CONVERT-TEST] ✅ Edge cases validated");
        return true;
    }

    /**
     * Test 8: Performance bounds - ensure optimizations don't cause extreme slowdown
     */
    private static boolean testPerformanceBounds() {
        logger.info("[GPU-CONVERT-TEST] Testing performance bounds...");

        int testSize = 100000;
        MockFloatTensor tensor = createTestTensor(testSize);

        // Baseline timing with all optimizations disabled
        disableAllOptimizations();
        long baselineStart = System.nanoTime();
        GPUTensorConverter.convertToHalfFloatArray(tensor);
        long baselineTime = System.nanoTime() - baselineStart;

        // Test with all optimizations enabled
        enableAllOptimizations();
        long optimizedStart = System.nanoTime();
        GPUTensorConverter.convertToHalfFloatArray(tensor);
        long optimizedTime = System.nanoTime() - optimizedStart;

        double speedupRatio = (double) baselineTime / optimizedTime;

        logger.info(String.format("[GPU-CONVERT-TEST] Performance ratio: %.2fx (baseline: %d ms, optimized: %d ms)",
                                 speedupRatio, baselineTime / 1_000_000, optimizedTime / 1_000_000));

        // Optimizations should provide speedup, but allow for some variance
        if (speedupRatio < 0.5) { // Less than 50% of baseline speed is problematic
            logger.severe(String.format("[GPU-CONVERT-TEST] Performance degradation detected: %.2fx slower", 1.0/speedupRatio));
            return false;
        }

        if (speedupRatio > 1.2) {
            logger.info(String.format("[GPU-CONVERT-TEST] ✅ Performance improvement: %.2fx speedup", speedupRatio));
        } else {
            logger.info("[GPU-CONVERT-TEST] ✅ Performance maintained within acceptable bounds");
        }

        return true;
    }

    /**
     * Compare two HalfFloatArrays for equality within epsilon tolerance
     */
    private static boolean compareHalfFloatArrays(HalfFloatArray expected, HalfFloatArray actual,
                                                String testName, int size) {
        if (expected.getSize() != actual.getSize()) {
            logger.severe(String.format("[GPU-CONVERT-TEST] %s: Size mismatch - expected %d, got %d",
                                       testName, expected.getSize(), actual.getSize()));
            return false;
        }

        int mismatches = 0;
        float maxDiff = 0.0f;

        for (int i = 0; i < expected.getSize(); i++) {
            float expectedVal = expected.get(i).getFloat32();
            float actualVal = actual.get(i).getFloat32();
            float diff = Math.abs(expectedVal - actualVal);

            if (diff > EPSILON) {
                mismatches++;
                maxDiff = Math.max(maxDiff, diff);

                if (mismatches <= 5) { // Report first 5 mismatches
                    logger.warning(String.format("[GPU-CONVERT-TEST] %s: Mismatch at index %d - expected %.6f, got %.6f (diff: %.6f)",
                                                testName, i, expectedVal, actualVal, diff));
                }
            }
        }

        if (mismatches > 0) {
            logger.severe(String.format("[GPU-CONVERT-TEST] %s: %d/%d elements mismatched (%.2f%%), max diff: %.6f",
                                       testName, mismatches, size, (100.0 * mismatches) / size, maxDiff));
            return false;
        }

        logger.info(String.format("[GPU-CONVERT-TEST] %s: All %d elements match within tolerance", testName, size));
        return true;
    }

    /**
     * Create test tensor with random data
     */
    private static MockFloatTensor createTestTensor(int size) {
        float[] data = new float[size];
        ThreadLocalRandom random = ThreadLocalRandom.current();

        for (int i = 0; i < size; i++) {
            // Generate diverse test data including edge cases
            if (i % 1000 == 0) {
                // Inject some special values
                data[i] = random.nextBoolean() ? 65504.0f : -65504.0f; // HalfFloat limits
            } else if (i % 500 == 0) {
                data[i] = 0.0f;
            } else {
                data[i] = random.nextFloat() * 200.0f - 100.0f; // Range [-100, 100]
            }
        }

        return new MockFloatTensor(data);
    }

    /**
     * Disable all optimizations for baseline testing
     */
    private static void disableAllOptimizations() {
        System.setProperty("gpu.tensor.conversion", "false");
        System.setProperty("gpu.tensor.conversion.parallel.cpu", "false");
        System.setProperty("gpu.tensor.conversion.bulk.copy", "false");
        System.setProperty("gpu.tensor.conversion.preprocessing", "false");
        System.setProperty("gpu.tensor.conversion.streaming", "false");
        System.setProperty("gpu.tensor.conversion.pool.buffers", "false");
        System.setProperty("gpu.tensor.conversion.unsafe.bulk", "false");
    }

    /**
     * Enable all optimizations for performance testing
     */
    private static void enableAllOptimizations() {
        System.setProperty("gpu.tensor.conversion", "true");
        System.setProperty("gpu.tensor.conversion.parallel.cpu", "true");
        System.setProperty("gpu.tensor.conversion.bulk.copy", "true");
        System.setProperty("gpu.tensor.conversion.preprocessing", "true");
        System.setProperty("gpu.tensor.conversion.streaming", "true");
        System.setProperty("gpu.tensor.conversion.pool.buffers", "true");
        System.setProperty("gpu.tensor.conversion.unsafe.bulk", "true");
    }

    /**
     * Mock FloatTensor implementation for testing
     */
    private static class MockFloatTensor extends FloatTensor {
        private final float[] data;

        public MockFloatTensor(float[] data) {
            this.data = data.clone();
        }

        @Override
        public float getFloat(int index) {
            return data[index];
        }

        @Override
        public void setFloat(int index, float value) {
            data[index] = value;
        }

        @Override
        public int size() {
            return data.length;
        }

        @Override
        public java.lang.foreign.MemorySegment asMemorySegment() {
            return null; // No memory segment for testing
        }

        @Override
        protected GGMLType type() {
            return GGMLType.F32; // Mock tensor uses F32 data
        }

        @Override
        protected FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
            // Simple implementation for testing - load from array
            int vectorLength = species.length();
            float[] vectorData = new float[vectorLength];

            for (int i = 0; i < vectorLength; i++) {
                int index = offset + i;
                vectorData[i] = (index < data.length) ? data[index] : 0.0f;
            }

            return FloatVector.fromArray(species, vectorData, 0);
        }

        @Override
        public String toString() {
            return String.format("MockFloatTensor[%d]", data.length);
        }
    }

    /**
     * Main method for standalone testing
     */
    public static void main(String[] args) {
        System.out.println("GPU Tensor Converter Correctness Test Suite");
        System.out.println("===========================================");

        boolean success = runAllTests();

        System.out.println("\n" + (success ? "✅ ALL TESTS PASSED" : "❌ TESTS FAILED"));
        System.exit(success ? 0 : 1);
    }
}