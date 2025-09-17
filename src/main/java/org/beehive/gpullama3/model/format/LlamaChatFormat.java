package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.LlamaTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LlamaChatFormat implements ChatFormat {

    protected final LlamaTokenizer tokenizer;
    protected final int beginOfText;
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final Set<Integer> stopTokens;

    public LlamaChatFormat(LlamaTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        
        // Handle both Llama-2 and Llama-3 based models
        // LLaVA-1.5-7B is based on Llama-2 and may not have Llama-3 specific tokens
        System.out.println("DEBUG: Available special tokens: " + specialTokens.size());
        specialTokens.forEach((key, value) -> System.out.println("  " + key + " = " + value));
        
        // Try Llama-3 tokens first, fallback to Llama-2 or safe defaults
        this.beginOfText = specialTokens.getOrDefault("<|begin_of_text|>", 1); // BOS token
        this.startHeader = specialTokens.getOrDefault("<|start_header_id|>", -1);
        this.endHeader = specialTokens.getOrDefault("<|end_header_id|>", -1);  
        this.endOfTurn = specialTokens.getOrDefault("<|eot_id|>", 2); // EOS token fallback
        this.endOfText = specialTokens.getOrDefault("<|end_of_text|>", 2); // EOS token fallback
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        
        // Use available tokens for stop conditions - handle duplicates
        if (this.endOfText != -1 && this.endOfTurn != -1 && this.endOfText != this.endOfTurn) {
            this.stopTokens = Set.of(endOfText, endOfTurn);
        } else if (this.endOfText != -1) {
            this.stopTokens = Set.of(endOfText);
        } else if (this.endOfTurn != -1) {
            this.stopTokens = Set.of(endOfTurn);
        } else {
            // Fallback to standard EOS token
            this.stopTokens = Set.of(2); // Standard EOS token ID
        }
        
        System.out.println("DEBUG: Chat format initialized with tokens:");
        System.out.println("  beginOfText: " + beginOfText);
        System.out.println("  endOfText: " + endOfText);  
        System.out.println("  endOfTurn: " + endOfTurn);
        System.out.println("  stopTokens: " + stopTokens);
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(encodeHeader(new Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }
}