package org.beehive.gpullama3.vision.encoder;

import org.beehive.gpullama3.multimodal.data.ImageData;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Interface for vision encoders that process images into feature representations.
 * Supports different vision architectures (SigLIP2, MoonViT, etc.)
 */
public interface VisionEncoder {
    
    /**
     * Encode image data into feature vectors.
     * 
     * @param image Preprocessed image data
     * @return Feature representation as FloatArray for GPU compatibility
     */
    FloatArray encode(ImageData image);
    
    /**
     * Encode image with native resolution support (for models that support variable resolution).
     * 
     * @param image Preprocessed image data with original aspect ratio
     * @return Feature representation with variable token count
     */
    default FloatArray encodeNativeResolution(ImageData image) {
        return encode(image); // Default to standard encoding
    }
    
    /**
     * Get the output feature dimension of this encoder.
     * 
     * @return Dimension of encoded features
     */
    int getFeatureDimension();
    
    /**
     * Get the number of tokens produced for a standard image.
     * 
     * @return Number of visual tokens
     */
    int getTokenCount();
    
    /**
     * Get encoder-specific configuration information.
     * 
     * @return Configuration string for debugging/logging
     */
    String getEncoderInfo();
    
    /**
     * Check if this encoder supports native resolution processing.
     * 
     * @return True if native resolution is supported
     */
    default boolean supportsNativeResolution() {
        return false;
    }
    
    /**
     * Release any resources held by this encoder.
     */
    void close();
}