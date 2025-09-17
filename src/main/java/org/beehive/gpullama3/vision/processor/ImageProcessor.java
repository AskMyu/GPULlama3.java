package org.beehive.gpullama3.vision.processor;

import org.beehive.gpullama3.multimodal.data.ImageData;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.Image;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

/**
 * Image processing pipeline for Vision Language Models.
 * Handles image loading, preprocessing, normalization, and patch extraction.
 */
public class ImageProcessor {
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};
    
    private final NDManager manager;
    private final ImageFactory imageFactory;
    
    public ImageProcessor() {
        this.manager = NDManager.newBaseManager();
        this.imageFactory = ImageFactory.getInstance();
    }
    
    /**
     * Process image bytes into normalized ImageData for VLM input.
     * 
     * @param imageBytes Raw image bytes (JPEG/PNG)
     * @param targetSize Target square size for resizing
     * @param preserveAspectRatio Whether to preserve original aspect ratio
     * @return Processed ImageData ready for vision encoder
     */
    public ImageData preprocessImage(byte[] imageBytes, int targetSize, boolean preserveAspectRatio) {
        try {
            // Load image using DJL OpenCV
            Image image = imageFactory.fromInputStream(new ByteArrayInputStream(imageBytes));
            
            // Resize image
            Image resizedImage = preserveAspectRatio 
                ? resizePreservingAspectRatio(image, targetSize)
                : resizeSquare(image, targetSize);
            
            // Convert to tensor and normalize  
            NDArray tensor = resizedImage.toNDArray(manager, Image.Flag.COLOR);
            tensor = normalizeImageNet(tensor);
            
            // Convert NDArray to ImageData
            return ndArrayToImageData(tensor);
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to preprocess image", e);
        }
    }
    
    /**
     * Extract patches from image for Vision Transformer processing.
     * 
     * @param image Source image data
     * @param patchSize Size of each square patch
     * @return Flattened patch data [num_patches, patch_size*patch_size*channels]
     */
    public float[][] extractPatches(ImageData image, int patchSize) {
        int height = image.getHeight();
        int width = image.getWidth();
        int channels = image.getChannels();
        float[][][] pixels = image.getPixels();
        
        int patchesY = height / patchSize;
        int patchesX = width / patchSize;
        int numPatches = patchesY * patchesX;
        int patchElements = patchSize * patchSize * channels;
        
        float[][] patches = new float[numPatches][patchElements];
        
        int patchIndex = 0;
        for (int py = 0; py < patchesY; py++) {
            for (int px = 0; px < patchesX; px++) {
                extractSinglePatch(pixels, py * patchSize, px * patchSize, 
                                 patchSize, channels, patches[patchIndex++]);
            }
        }
        
        return patches;
    }
    
    /**
     * Calculate optimal image processing parameters for different models.
     */
    public static class ImageConfig {
        public final int imageSize;
        public final int patchSize;
        public final int numPatches;
        
        public ImageConfig(int imageSize, int patchSize) {
            this.imageSize = imageSize;
            this.patchSize = patchSize;
            this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        }
        
        // Standard configurations for common vision models
        public static final ImageConfig STANDARD_384 = new ImageConfig(384, 14);
        public static final ImageConfig HIGH_RES_512 = new ImageConfig(512, 16);
        public static final ImageConfig ULTRA_HIGH_RES_1792 = new ImageConfig(1792, 14);
    }
    
    private Image resizeSquare(Image image, int size) {
        return image.resize(size, size, false);
    }
    
    private Image resizePreservingAspectRatio(Image image, int maxSize) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        float scale = Math.min((float) maxSize / width, (float) maxSize / height);
        int newWidth = Math.round(width * scale);
        int newHeight = Math.round(height * scale);
        
        return image.resize(newWidth, newHeight, false);
    }
    
    private NDArray normalizeImageNet(NDArray tensor) {
        // Convert to float and normalize to [0, 1]
        tensor = tensor.div(255.0f);
        
        // Apply ImageNet normalization
        NDArray mean = manager.create(IMAGENET_MEAN).reshape(3, 1, 1);
        NDArray std = manager.create(IMAGENET_STD).reshape(3, 1, 1);
        
        return tensor.sub(mean).div(std);
    }
    
    private ImageData ndArrayToImageData(NDArray tensor) {
        // Expect tensor in CHW format, convert to HWC for ImageData
        long[] shape = tensor.getShape().getShape();
        int channels = (int) shape[0];
        int height = (int) shape[1];
        int width = (int) shape[2];
        
        float[] flatData = tensor.toFloatArray();
        float[][][] pixels = new float[height][width][channels];
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int flatIndex = c * height * width + h * width + w;
                    pixels[h][w][c] = flatData[flatIndex];
                }
            }
        }
        
        return new ImageData(pixels, width, height);
    }
    
    private void extractSinglePatch(float[][][] pixels, int startY, int startX, 
                                   int patchSize, int channels, float[] patchData) {
        int index = 0;
        for (int y = startY; y < startY + patchSize; y++) {
            for (int x = startX; x < startX + patchSize; x++) {
                for (int c = 0; c < channels; c++) {
                    patchData[index++] = pixels[y][x][c];
                }
            }
        }
    }
    
    public void close() {
        manager.close();
    }
}