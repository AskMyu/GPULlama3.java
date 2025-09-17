package org.beehive.gpullama3.vision.processor;

import org.beehive.gpullama3.multimodal.data.ImageData;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Native Java image processing pipeline for Vision Language Models.
 * Implements CLIP ViT preprocessing without DJL dependencies.
 * 
 * Uses correct CLIP normalization values (not ImageNet):
 * - Mean: [0.48145466, 0.4578275, 0.40821073] 
 * - Std: [0.26862954, 0.26130258, 0.27577711]
 */
public class NativeImageProcessor {
    
    // CLIP-specific normalization parameters (NOT ImageNet values)
    private static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};
    
    /**
     * Process image bytes into normalized ImageData for CLIP ViT models.
     * 
     * @param imageBytes Raw image bytes (JPEG/PNG)
     * @param targetSize Target square size for resizing (typically 336 for CLIP ViT-L)
     * @param preserveAspectRatio Whether to preserve original aspect ratio with center crop
     * @return Processed ImageData ready for vision encoder
     */
    public ImageData preprocessImage(byte[] imageBytes, int targetSize, boolean preserveAspectRatio) {
        try {
            // Load image using standard Java ImageIO
            BufferedImage originalImage = ImageIO.read(new ByteArrayInputStream(imageBytes));
            if (originalImage == null) {
                throw new IllegalArgumentException("Unable to decode image from provided bytes");
            }
            
            // Resize image using bicubic interpolation 
            BufferedImage resizedImage = preserveAspectRatio 
                ? resizeWithAspectRatio(originalImage, targetSize)
                : resizeSquare(originalImage, targetSize);
            
            // Convert to normalized float pixels with CLIP parameters
            float[][][] normalizedPixels = normalizeImageCLIP(resizedImage);
            
            // Create ImageData with original bytes for caching
            return new ImageData(normalizedPixels, resizedImage.getWidth(), resizedImage.getHeight(), imageBytes);
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to preprocess image", e);
        } catch (Exception e) {
            throw new RuntimeException("Image processing error: " + e.getMessage(), e);
        }
    }
    
    /**
     * Resize image while preserving aspect ratio and center cropping to square.
     * Uses bicubic interpolation for high quality scaling.
     */
    private BufferedImage resizeWithAspectRatio(BufferedImage original, int targetSize) {
        int originalWidth = original.getWidth();
        int originalHeight = original.getHeight();
        
        // Calculate scaling to make shorter side equal to targetSize
        float scale = (float) targetSize / Math.min(originalWidth, originalHeight);
        
        int scaledWidth = Math.round(originalWidth * scale);
        int scaledHeight = Math.round(originalHeight * scale);
        
        // First resize with bicubic interpolation
        BufferedImage scaled = resizeHighQuality(original, scaledWidth, scaledHeight);
        
        // Then center crop to square
        int cropX = (scaledWidth - targetSize) / 2;
        int cropY = (scaledHeight - targetSize) / 2;
        
        BufferedImage cropped = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = cropped.createGraphics();
        
        // Use high quality rendering
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        
        g2d.drawImage(scaled, 0, 0, targetSize, targetSize, 
                      cropX, cropY, cropX + targetSize, cropY + targetSize, null);
        g2d.dispose();
        
        return cropped;
    }
    
    /**
     * Resize image to exact square dimensions using bicubic interpolation.
     */
    private BufferedImage resizeSquare(BufferedImage original, int targetSize) {
        return resizeHighQuality(original, targetSize, targetSize);
    }
    
    /**
     * High-quality image resizing using bicubic interpolation.
     */
    private BufferedImage resizeHighQuality(BufferedImage original, int targetWidth, int targetHeight) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        
        // Enable high-quality rendering
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        g2d.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        
        return resized;
    }
    
    /**
     * Normalize image pixels using CLIP-specific parameters.
     * 
     * CLIP preprocessing steps:
     * 1. Convert pixels to 0-1 range (divide by 255)
     * 2. Subtract CLIP mean per channel: [0.48145466, 0.4578275, 0.40821073]
     * 3. Divide by CLIP std per channel: [0.26862954, 0.26130258, 0.27577711]
     * 
     * @param image Input BufferedImage (RGB format)
     * @return Normalized pixel array [height][width][channels] 
     */
    private float[][][] normalizeImageCLIP(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[][][] pixels = new float[height][width][3]; // RGB channels
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                
                // Extract RGB components (0-255 range)
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                
                // Convert to 0-1 range
                float rNorm = r / 255.0f;
                float gNorm = g / 255.0f;
                float bNorm = b / 255.0f;
                
                // Apply CLIP normalization: (pixel - mean) / std
                pixels[y][x][0] = (rNorm - CLIP_MEAN[0]) / CLIP_STD[0]; // Red channel
                pixels[y][x][1] = (gNorm - CLIP_MEAN[1]) / CLIP_STD[1]; // Green channel  
                pixels[y][x][2] = (bNorm - CLIP_MEAN[2]) / CLIP_STD[2]; // Blue channel
            }
        }
        
        return pixels;
    }
    
    /**
     * Extract patches from image for Vision Transformer processing.
     * 
     * @param image Source image data
     * @param patchSize Size of each square patch (typically 14 for CLIP ViT)
     * @return Flattened patch data [num_patches, patch_size*patch_size*channels]
     */
    public float[][] extractPatches(ImageData image, int patchSize) {
        int height = image.getHeight();
        int width = image.getWidth();
        int channels = image.getChannels();
        float[][][] pixels = image.getPixels();
        
        // Calculate number of patches
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
     * Extract a single patch from the image.
     */
    private void extractSinglePatch(float[][][] pixels, int startY, int startX, 
                                   int patchSize, int channels, float[] patch) {
        int index = 0;
        
        for (int y = startY; y < startY + patchSize; y++) {
            for (int x = startX; x < startX + patchSize; x++) {
                for (int c = 0; c < channels; c++) {
                    patch[index++] = pixels[y][x][c];
                }
            }
        }
    }
    
    /**
     * Get image processing configuration for different CLIP models.
     */
    public static class CLIPImageConfig {
        public final int imageSize;
        public final int patchSize;
        public final int numPatches;
        
        public CLIPImageConfig(int imageSize, int patchSize) {
            this.imageSize = imageSize;
            this.patchSize = patchSize;
            this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        }
        
        // CLIP ViT-Large/14@336px (used in LLaVA-Llama-3-8B)
        public static final CLIPImageConfig CLIP_VIT_L_336 = new CLIPImageConfig(336, 14);
        
        // CLIP ViT-Base/32@224px  
        public static final CLIPImageConfig CLIP_VIT_B_224 = new CLIPImageConfig(224, 32);
        
        @Override
        public String toString() {
            return String.format("CLIPImageConfig{imageSize=%d, patchSize=%d, numPatches=%d}", 
                               imageSize, patchSize, numPatches);
        }
    }
}