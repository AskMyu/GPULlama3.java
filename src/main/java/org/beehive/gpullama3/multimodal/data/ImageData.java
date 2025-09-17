package org.beehive.gpullama3.multimodal.data;

/**
 * Represents processed image data for VLM input.
 * Contains normalized pixel data and metadata.
 */
public class ImageData {
    private final float[][][] pixels; // [height][width][channels]
    private final int width;
    private final int height;
    private final int channels;
    private final byte[] originalBytes; // Original image bytes for caching
    
    public ImageData(float[][][] pixels, int width, int height) {
        this(pixels, width, height, null);
    }
    
    public ImageData(float[][][] pixels, int width, int height, byte[] originalBytes) {
        this.pixels = pixels;
        this.width = width;
        this.height = height;
        this.channels = pixels.length > 0 && pixels[0].length > 0 ? pixels[0][0].length : 3;
        this.originalBytes = originalBytes;
    }
    
    public float[][][] getPixels() {
        return pixels;
    }
    
    public int getWidth() {
        return width;
    }
    
    public int getHeight() {
        return height;
    }
    
    public int getChannels() {
        return channels;
    }
    
    public float[] getFlattenedPixels() {
        float[] flattened = new float[height * width * channels];
        int index = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    flattened[index++] = pixels[h][w][c];
                }
            }
        }
        return flattened;
    }
    
    public int getTotalPixels() {
        return height * width * channels;
    }
    
    /**
     * Get original image bytes for caching purposes.
     * @return Original image bytes or null if not available
     */
    public byte[] getOriginalBytes() {
        return originalBytes;
    }
    
    @Override
    public String toString() {
        return String.format("ImageData{width=%d, height=%d, channels=%d, totalPixels=%d}", 
                           width, height, channels, getTotalPixels());
    }
}