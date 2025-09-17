package org.beehive.gpullama3.vision.encoder;

import org.beehive.gpullama3.multimodal.data.ImageData;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

import ai.onnxruntime.*;
import java.util.Map;
import java.util.Collections;
import java.nio.FloatBuffer;

/**
 * SigLIP2 Vision Encoder using ONNX Runtime for inference.
 * Supports multiple SigLIP2 variants: Base, Large, So400m, Giant.
 */
public class SigLIP2Encoder implements VisionEncoder {
    
    public enum SigLIP2Variant {
        BASE(86_000_000, 768, "siglip2-base"),
        LARGE(303_000_000, 1024, "siglip2-large"), 
        SO400M(400_000_000, 1152, "siglip2-so400m"),
        GIANT(1_000_000_000, 1408, "siglip2-giant");
        
        public final int parameters;
        public final int featureDim;
        public final String modelName;
        
        SigLIP2Variant(int parameters, int featureDim, String modelName) {
            this.parameters = parameters;
            this.featureDim = featureDim;
            this.modelName = modelName;
        }
    }
    
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final SigLIP2Variant variant;
    private final boolean nativeResolutionSupported;
    private final int standardTokenCount;
    
    /**
     * Initialize SigLIP2 encoder with ONNX model file.
     * 
     * @param modelPath Path to ONNX model file
     * @param variant SigLIP2 model variant
     * @param nativeResolution Whether this model supports native resolution
     */
    public SigLIP2Encoder(String modelPath, SigLIP2Variant variant, boolean nativeResolution) {
        try {
            this.environment = OrtEnvironment.getEnvironment();
            this.session = environment.createSession(modelPath, new OrtSession.SessionOptions());
            this.variant = variant;
            this.nativeResolutionSupported = nativeResolution;
            this.standardTokenCount = calculateStandardTokenCount();
            
            System.out.println("SigLIP2 Encoder initialized: " + getEncoderInfo());
            
        } catch (OrtException e) {
            throw new RuntimeException("Failed to initialize SigLIP2 encoder", e);
        }
    }
    
    /**
     * Convenience constructor for So400m variant (balanced performance/efficiency).
     */
    public static SigLIP2Encoder createSo400m(String modelPath) {
        return new SigLIP2Encoder(modelPath, SigLIP2Variant.SO400M, false);
    }
    
    /**
     * Constructor for native resolution variant (NaFlex).
     */
    public static SigLIP2Encoder createNaFlex(String modelPath, SigLIP2Variant variant) {
        return new SigLIP2Encoder(modelPath, variant, true);
    }
    
    @Override
    public FloatArray encode(ImageData image) {
        try {
            // Prepare input tensor from ImageData
            float[] imagePixels = image.getFlattenedPixels();
            long[] inputShape = {1, image.getChannels(), image.getHeight(), image.getWidth()};
            
            // Create ONNX tensor
            OnnxTensor inputTensor = OnnxTensor.createTensor(environment, 
                FloatBuffer.wrap(imagePixels), inputShape);
            
            // Run inference
            Map<String, OnnxTensorLike> inputs = Collections.singletonMap("pixel_values", inputTensor);
            OrtSession.Result results = session.run(inputs);
            
            // Extract features
            OnnxValue output = results.get("pooler_output").orElse(
                results.get("image_embeds").orElse(
                    results.get(0) // Fallback to first output
                )
            );
            
            float[][] features = (float[][]) output.getValue();
            
            // Convert to FloatArray for TornadoVM compatibility
            FloatArray featureArray = new FloatArray(features[0].length);
            for (int i = 0; i < features[0].length; i++) {
                featureArray.set(i, features[0][i]);
            }
            
            // Cleanup
            inputTensor.close();
            results.close();
            
            return featureArray;
            
        } catch (OrtException e) {
            throw new RuntimeException("SigLIP2 encoding failed", e);
        }
    }
    
    @Override
    public FloatArray encodeNativeResolution(ImageData image) {
        if (!nativeResolutionSupported) {
            return encode(image); // Fallback to standard encoding
        }
        
        // For NaFlex variant, calculate optimal sequence length
        int sequenceLength = calculateOptimalSequenceLength(image);
        return encodeWithSequenceLength(image, sequenceLength);
    }
    
    @Override
    public int getFeatureDimension() {
        return variant.featureDim;
    }
    
    @Override
    public int getTokenCount() {
        return standardTokenCount;
    }
    
    @Override
    public String getEncoderInfo() {
        return String.format("SigLIP2-%s (%d params, %d features, %d tokens, native_res=%b)",
            variant.name(), variant.parameters, variant.featureDim, 
            standardTokenCount, nativeResolutionSupported);
    }
    
    @Override
    public boolean supportsNativeResolution() {
        return nativeResolutionSupported;
    }
    
    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            System.err.println("Error closing SigLIP2 encoder: " + e.getMessage());
        }
    }
    
    private int calculateStandardTokenCount() {
        // Standard calculation for vision transformers
        // Assumes 224x224 input with 16x16 patches = 196 tokens + 1 class token
        // For SigLIP2, may vary by variant
        switch (variant) {
            case BASE:
            case LARGE:
                return 197; // 14x14 patches + class token
            case SO400M:
                return 256; // Higher resolution support
            case GIANT:
                return 577; // 24x24 patches + class token
            default:
                return 197;
        }
    }
    
    private int calculateOptimalSequenceLength(ImageData image) {
        // NaFlex variant supports: 128, 256, 576, 784, 1024 tokens
        int[] supportedLengths = {128, 256, 576, 784, 1024};
        
        // Calculate based on aspect ratio and resolution
        double aspectRatio = (double) image.getWidth() / image.getHeight();
        int pixels = image.getWidth() * image.getHeight();
        
        // Select appropriate sequence length based on image characteristics
        if (pixels <= 224 * 224) return 128;
        if (pixels <= 384 * 384) return 256;
        if (pixels <= 512 * 512) return 576;
        if (pixels <= 768 * 768) return 784;
        return 1024;
    }
    
    private FloatArray encodeWithSequenceLength(ImageData image, int sequenceLength) {
        // For now, delegate to standard encoding
        // Implementation pending: sequence-length-specific processing for NaFlex
        return encode(image);
    }
}