package org.beehive.gpullama3.multimodal.chat;

import org.beehive.gpullama3.multimodal.data.ImageData;
import org.beehive.gpullama3.multimodal.data.MultimodalInput;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.vision.processor.ImageProcessor;

import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.Base64;

/**
 * Handles multimodal chat format parsing and tokenization.
 * Supports standard OpenAI-style vision chat formats with image URLs and base64.
 */
public class MultimodalChatFormat {
    public static final String IMAGE_TOKEN = "<|image|>";
    public static final String IMAGE_PLACEHOLDER = "<image>";
    public static final int IMAGE_TOKEN_ID = 32000; // Special token ID for images
    
    private static final Pattern BASE64_IMAGE_PATTERN = 
        Pattern.compile("data:image/(?:jpeg|jpg|png|gif|webp);base64,([A-Za-z0-9+/=]+)");
    
    private final ImageProcessor imageProcessor;
    private final Tokenizer textTokenizer;
    private final Set<String> allowedSpecialTokens;
    
    public MultimodalChatFormat(Tokenizer textTokenizer) {
        this.textTokenizer = textTokenizer;
        this.imageProcessor = new ImageProcessor();
        this.allowedSpecialTokens = new HashSet<>();
        this.allowedSpecialTokens.add(IMAGE_TOKEN);
    }
    
    /**
     * Parse a multimodal chat message into MultimodalInput.
     * Supports various input formats:
     * - Text with <image> placeholders
     * - JSON-style content arrays
     * - Base64 encoded images
     * 
     * @param messageText Text content potentially containing image references
     * @param imageUrls List of image URLs or base64 data URIs
     * @param imageSize Target size for image processing
     * @return Processed MultimodalInput ready for VLM
     */
    public MultimodalInput parseMessage(String messageText, List<String> imageUrls, int imageSize) {
        List<ImageData> images = new ArrayList<>();
        String processedText = messageText;
        
        // Process images from URLs
        for (String imageUrl : imageUrls) {
            try {
                ImageData image = loadImageFromUrl(imageUrl, imageSize);
                images.add(image);
                
                // Replace first occurrence of placeholder with image token
                processedText = processedText.replaceFirst(
                    Pattern.quote(IMAGE_PLACEHOLDER), IMAGE_TOKEN);
                
            } catch (Exception e) {
                System.err.println("Failed to load image: " + imageUrl + " - " + e.getMessage());
            }
        }
        
        // Tokenize the processed text
        int[] textTokens = tokenizeWithImagePlaceholders(processedText, images.size());
        
        return new MultimodalInput(processedText, images, textTokens);
    }
    
    /**
     * Parse simple text + single image input.
     */
    public MultimodalInput parseTextAndImage(String text, String imageUrl, int imageSize) {
        return parseMessage(text, List.of(imageUrl), imageSize);
    }
    
    /**
     * Parse text-only input (for compatibility).
     */
    public MultimodalInput parseTextOnly(String text) {
        List<Integer> tokenList = textTokenizer.encode(text, allowedSpecialTokens);
        int[] tokens = tokenList.stream().mapToInt(Integer::intValue).toArray();
        return MultimodalInput.textOnly(text, tokens);
    }
    
    /**
     * Tokenize text with image token placeholders.
     * Replaces <|image|> tokens with special image token IDs and calculates
     * the appropriate number of visual tokens based on the model.
     * 
     * @param text Text with image token placeholders
     * @param imageCount Number of images in the input
     * @return Token array with image tokens properly placed
     */
    public int[] tokenizeWithImagePlaceholders(String text, int imageCount) {
        if (imageCount == 0) {
            List<Integer> tokenList = textTokenizer.encode(text, allowedSpecialTokens);
            return tokenList.stream().mapToInt(Integer::intValue).toArray();
        }
        
        List<Integer> tokens = new ArrayList<>();
        String[] segments = text.split(Pattern.quote(IMAGE_TOKEN), -1);
        
        for (int i = 0; i < segments.length; i++) {
            // Add text tokens for this segment
            if (!segments[i].isEmpty()) {
                List<Integer> segmentTokens = textTokenizer.encode(segments[i], allowedSpecialTokens);
                for (int token : segmentTokens) {
                    tokens.add(token);
                }
            }
            
            // Add image tokens if not the last segment and we have images
            if (i < segments.length - 1 && i < imageCount) {
                int imageTokenCount = getImageTokenCount();
                for (int j = 0; j < imageTokenCount; j++) {
                    tokens.add(IMAGE_TOKEN_ID);
                }
            }
        }
        
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * Get the number of tokens each image should produce.
     * This depends on the specific VLM architecture.
     */
    protected int getImageTokenCount() {
        // Default vision token count: 81 tokens per image (384x384 -> 14x14 patches + class token)
        return 81;
    }
    
    /**
     * Configure image token count for different VLM models.
     */
    public static class ModelConfig {
        public final int imageTokens;
        public final int imageSize;
        public final String modelName;
        
        public ModelConfig(int imageTokens, int imageSize, String modelName) {
            this.imageTokens = imageTokens;
            this.imageSize = imageSize;
            this.modelName = modelName;
        }
        
        // Standard VLM configurations for different architectures
        public static final ModelConfig STANDARD_VLM_SMALL = new ModelConfig(81, 384, "Standard-VLM-Small");
        public static final ModelConfig STANDARD_VLM_MEDIUM = new ModelConfig(256, 512, "Standard-VLM-Medium");
        public static final ModelConfig HIGH_RES_VLM = new ModelConfig(577, 1792, "High-Resolution-VLM");
    }
    
    private ImageData loadImageFromUrl(String imageUrl, int imageSize) {
        try {
            byte[] imageBytes;
            
            // Check if it's a base64 data URI
            if (imageUrl.startsWith("data:image/")) {
                imageBytes = decodeBase64Image(imageUrl);
            } else {
                // For now, throw exception - URL loading would require HTTP client
                throw new UnsupportedOperationException(
                    "URL image loading not implemented. Use base64 data URIs.");
            }
            
            // Process image using DJL pipeline
            return imageProcessor.preprocessImage(imageBytes, imageSize, false);
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to load image: " + imageUrl, e);
        }
    }
    
    private byte[] decodeBase64Image(String dataUri) {
        Matcher matcher = BASE64_IMAGE_PATTERN.matcher(dataUri);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Invalid base64 image data URI");
        }
        
        String base64Data = matcher.group(1);
        return Base64.getDecoder().decode(base64Data);
    }
    
    public void close() {
        imageProcessor.close();
    }
    
    /**
     * Create format-specific chat handlers for different models.
     */
    public static MultimodalChatFormat createForModel(Tokenizer tokenizer, ModelConfig config) {
        return new ConfigurableMultimodalChatFormat(tokenizer, config);
    }
    
    /**
     * Configurable chat format that adapts to different model requirements.
     */
    private static class ConfigurableMultimodalChatFormat extends MultimodalChatFormat {
        private final ModelConfig config;
        
        public ConfigurableMultimodalChatFormat(Tokenizer tokenizer, ModelConfig config) {
            super(tokenizer);
            this.config = config;
        }
        
        @Override
        protected int getImageTokenCount() {
            return config.imageTokens;
        }
    }
}