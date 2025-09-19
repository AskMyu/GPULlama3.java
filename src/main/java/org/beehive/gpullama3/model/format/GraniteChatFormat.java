package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Tokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format handler for IBM Granite models.
 * 
 * Granite uses a structured conversation format with support for:
 * - Standard user/assistant conversation
 * - Structured reasoning with <think></think> tags
 * - Fill-in-the-Middle (FIM) operations for code completion
 */
public class GraniteChatFormat implements ChatFormat {
    
    private final Tokenizer tokenizer;
    
    // Granite conversation tokens (Granite 3.3 format)
    private static final String ROLE_START = "<|start_of_role|>";
    private static final String ROLE_END = "<|end_of_role|>";
    private static final String END_OF_TEXT = "<|end_of_text|>";
    
    public GraniteChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }
    
    @Override
    public List<Integer> encodeHeader(ChatFormat.Message message) {
        // For ASSISTANT role, we need to add the role header before generation starts
        if (message.role().equals(ChatFormat.Role.ASSISTANT)) {
            StringBuilder formatted = new StringBuilder();
            formatted.append(ROLE_START).append("assistant").append(ROLE_END);

            // Debug: Log the assistant header
            System.err.printf("[GRANITE-CHAT] Assistant header: '%s'%n", formatted.toString());

            List<Integer> tokens = tokenizer.encodeAsList(formatted.toString());
            System.err.printf("[GRANITE-CHAT] Assistant header encoded to %d tokens%n", tokens.size());
            return tokens;
        }

        // For other roles, Granite includes role information in the message itself
        return new ArrayList<>();
    }
    
    @Override
    public List<Integer> encodeMessage(ChatFormat.Message message) {
        StringBuilder formatted = new StringBuilder();

        // Format based on role using Granite 3.3 format
        if (message.role().equals(ChatFormat.Role.SYSTEM)) {
            formatted.append(ROLE_START).append("system").append(ROLE_END);
            formatted.append(message.content()).append(END_OF_TEXT);
        } else if (message.role().equals(ChatFormat.Role.USER)) {
            formatted.append(ROLE_START).append("user").append(ROLE_END);
            formatted.append(message.content()).append(END_OF_TEXT);
        } else if (message.role().equals(ChatFormat.Role.ASSISTANT)) {
            formatted.append(ROLE_START).append("assistant").append(ROLE_END);
            formatted.append(message.content()).append(END_OF_TEXT);
        } else if (message.role().equals(ChatFormat.Role.FIM_PREFIX)) {
            // Fill-in-the-Middle prefix
            formatted.append("<|fim_prefix|>");
            formatted.append(message.content());
        } else if (message.role().equals(ChatFormat.Role.FIM_SUFFIX)) {
            // Fill-in-the-Middle suffix
            formatted.append("<|fim_suffix|>");
            formatted.append(message.content());
            formatted.append("<|fim_middle|>"); // Signal for middle generation
        }
        
        // Debug: Log the formatted message
        System.err.printf("[GRANITE-CHAT] Formatted message: '%s'%n", formatted.toString());

        // Encode the formatted text
        List<Integer> tokens = tokenizer.encodeAsList(formatted.toString());
        System.err.printf("[GRANITE-CHAT] Encoded %d tokens for %s role%n", tokens.size(), message.role());
        return tokens;
    }
    
    @Override
    public int getBeginOfText() {
        // Return BOS token if available
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        return specialTokens.getOrDefault("<|begin_of_text|>",
               specialTokens.getOrDefault("<bos>", 1));
    }
    
    @Override
    public Set<Integer> getStopTokens() {
        Set<Integer> stopTokens = new HashSet<>();
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Add end of text and turn tokens as stop tokens
        if (specialTokens.containsKey("<|end_of_text|>")) {
            stopTokens.add(specialTokens.get("<|end_of_text|>"));
        }
        if (specialTokens.containsKey("<eos>")) {
            stopTokens.add(specialTokens.get("<eos>"));
        }
        if (specialTokens.containsKey("</think>")) {
            stopTokens.add(specialTokens.get("</think>"));
        }
        if (specialTokens.containsKey("<|fim_middle|>")) {
            stopTokens.add(specialTokens.get("<|fim_middle|>"));
        }

        // CRITICAL FIX: Ensure token 0 is NOT treated as a stop token for Granite models
        // Token 0 appears to be incorrectly mapped in some GGUF files
        stopTokens.remove(0);

        System.err.printf("[GRANITE-CHAT] Stop tokens configured: %s%n", stopTokens);

        return stopTokens;
    }
    
    /**
     * Creates a prompt for Fill-in-the-Middle code completion.
     * 
     * @param prefix The code before the cursor
     * @param suffix The code after the cursor
     * @return Encoded tokens for FIM completion
     */
    public List<Integer> encodeFIMPrompt(String prefix, String suffix) {
        List<Integer> tokens = new ArrayList<>();
        
        // Add BOS token
        tokens.add(getBeginOfText());
        
        // Add FIM prefix
        tokens.addAll(encodeMessage(new ChatFormat.Message(ChatFormat.Role.FIM_PREFIX, prefix)));
        
        // Add FIM suffix (this also adds the middle token)
        tokens.addAll(encodeMessage(new ChatFormat.Message(ChatFormat.Role.FIM_SUFFIX, suffix)));
        
        return tokens;
    }
}