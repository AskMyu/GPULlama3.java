package org.beehive.gpullama3.multimodal.data;

import java.util.List;
import java.util.ArrayList;

/**
 * Represents multimodal input combining text and images for VLM processing.
 * Supports interleaved text-image conversations.
 */
public class MultimodalInput {
    private final List<String> textSegments;
    private final List<ImageData> images;
    private final String combinedText;
    private final int[] textTokens;
    
    public MultimodalInput(List<String> textSegments, List<ImageData> images) {
        this.textSegments = new ArrayList<>(textSegments);
        this.images = new ArrayList<>(images);
        this.combinedText = String.join("", textSegments);
        this.textTokens = null; // Will be set during tokenization
    }
    
    public MultimodalInput(String text, List<ImageData> images, int[] textTokens) {
        this.textSegments = List.of(text);
        this.images = new ArrayList<>(images);
        this.combinedText = text;
        this.textTokens = textTokens;
    }
    
    public List<String> getTextSegments() {
        return new ArrayList<>(textSegments);
    }
    
    public List<ImageData> getImages() {
        return new ArrayList<>(images);
    }
    
    public String getCombinedText() {
        return combinedText;
    }
    
    public int[] getTextTokens() {
        return textTokens != null ? textTokens.clone() : null;
    }
    
    public boolean hasImages() {
        return !images.isEmpty();
    }
    
    public int getImageCount() {
        return images.size();
    }
    
    public ImageData getImage(int index) {
        if (index < 0 || index >= images.size()) {
            throw new IndexOutOfBoundsException("Image index out of bounds: " + index);
        }
        return images.get(index);
    }
    
    public boolean hasText() {
        return !combinedText.isEmpty();
    }
    
    /**
     * Create a text-only multimodal input.
     */
    public static MultimodalInput textOnly(String text, int[] tokens) {
        return new MultimodalInput(text, List.of(), tokens);
    }
    
    /**
     * Create an image-only multimodal input.
     */
    public static MultimodalInput imageOnly(List<ImageData> images) {
        return new MultimodalInput(List.of(), images);
    }
    
    /**
     * Create a simple text + single image input.
     */
    public static MultimodalInput textAndImage(String text, ImageData image, int[] tokens) {
        return new MultimodalInput(text, List.of(image), tokens);
    }
    
    @Override
    public String toString() {
        return String.format("MultimodalInput{textLength=%d, imageCount=%d, hasTokens=%b}",
                           combinedText.length(), images.size(), textTokens != null);
    }
    
    /**
     * Get a summary of the multimodal input for logging.
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("MultimodalInput{");
        sb.append("text=").append(combinedText.length()).append(" chars");
        sb.append(", images=").append(images.size());
        if (!images.isEmpty()) {
            sb.append(" [");
            for (int i = 0; i < images.size(); i++) {
                if (i > 0) sb.append(", ");
                ImageData img = images.get(i);
                sb.append(img.getWidth()).append("x").append(img.getHeight());
            }
            sb.append("]");
        }
        if (textTokens != null) {
            sb.append(", tokens=").append(textTokens.length);
        }
        sb.append("}");
        return sb.toString();
    }
}