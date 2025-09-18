package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Tokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format handler for Gemma models.
 * 
 * Gemma uses a simple conversational format with special tokens
 * to delineate between user and model turns.
 */
public class GemmaChatFormat implements ChatFormat {
    
    private final Tokenizer tokenizer;
    
    // Gemma conversation tokens
    private static final String USER_TOKEN = "<start_of_turn>user\n";
    private static final String MODEL_TOKEN = "<start_of_turn>model\n";
    private static final String END_TOKEN = "<end_of_turn>\n";
    
    public GemmaChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }
    
    @Override
    public List<Integer> encodeHeader(ChatFormat.Message message) {
        // Gemma doesn't use separate headers, return empty list
        return new ArrayList<>();
    }
    
    @Override
    public List<Integer> encodeMessage(ChatFormat.Message message) {
        StringBuilder formatted = new StringBuilder();
        
        // Format based on role
        if (message.role().equals(ChatFormat.Role.SYSTEM)) {
            // Gemma typically incorporates system messages as part of the first user message
            formatted.append(USER_TOKEN);
            formatted.append("System: ").append(message.content());
            formatted.append(END_TOKEN);
        } else if (message.role().equals(ChatFormat.Role.USER)) {
            formatted.append(USER_TOKEN);
            formatted.append(message.content());
            formatted.append(END_TOKEN);
        } else if (message.role().equals(ChatFormat.Role.ASSISTANT)) {
            formatted.append(MODEL_TOKEN);
            formatted.append(message.content());
            formatted.append(END_TOKEN);
        }
        
        // Encode the formatted text
        List<Integer> tokens = tokenizer.encodeAsList(formatted.toString());
        return tokens;
    }
    
    @Override
    public int getBeginOfText() {
        // Return BOS token if available
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        System.err.printf("[OLMOE-BOS-DEBUG] Available special tokens: %s%n", specialTokens);

        int bosToken = specialTokens.getOrDefault("<bos>", specialTokens.getOrDefault("<|bos|>", 1));
        System.err.printf("[OLMOE-BOS-DEBUG] Selected BOS token: %d%n", bosToken);

        return bosToken;
    }
    
    @Override
    public Set<Integer> getStopTokens() {
        Set<Integer> stopTokens = new HashSet<>();
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        
        // Add EOS token as stop token
        if (specialTokens.containsKey("<eos>")) {
            stopTokens.add(specialTokens.get("<eos>"));
        }
        if (specialTokens.containsKey("<|eos|>")) {
            stopTokens.add(specialTokens.get("<|eos|>"));
        }
        if (specialTokens.containsKey("<end_of_turn>")) {
            stopTokens.add(specialTokens.get("<end_of_turn>"));
        }
        
        return stopTokens;
    }
    
    @Override
    public ChatTokens chatTokens() {
        // Return Gemma-specific chat tokens
        return new ChatTokens(
            USER_TOKEN,           // tStartHeader: "<start_of_turn>user\n"  
            MODEL_TOKEN,          // tEndHeader: "<start_of_turn>model\n"
            END_TOKEN,            // tEndOfTurn: "<end_of_turn>\n"
            "<eos>",              // tEndOfText: "<eos>"
            "<eos>"               // tEndOfTextFim: "<eos>"
        );
    }
    
    // Note: encodeDialogue was removed as it's not part of the interface
}