package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.tornadovm.TornadoVMSafeInitializer;

/**
 * Vision-Language Model State implementation that extends LlamaState
 * to support multimodal inference with vision embeddings and text tokens.
 * 
 * This class provides the infrastructure for true multimodal fusion by:
 * - Storing vision embeddings alongside text representations
 * - Managing the position mapping between vision and text tokens
 * - Providing unified access to mixed embedding sources
 * 
 * Architecture:
 * - Inherits all text processing capabilities from LlamaState
 * - Adds vision embedding storage (typically 576 x 4096 for LLaVA)
 * - Manages sequence positioning for vision + text integration
 * - Enables direct embedding injection bypassing tokenization
 */
public final class VLMState extends LlamaState {
    
    // Vision-specific state components
    private FloatArray[] visionEmbeddings;           // Vision token embeddings [numVisionTokens x embedDim]
    private int[] visionTokenPositions;              // Positions where vision tokens appear in sequence
    private int numVisionTokens;                     // Number of vision tokens (e.g., 576 for LLaVA)
    private int textStartPosition;                   // Position where text tokens begin after vision
    private boolean hasVisionEmbeddings;             // Flag indicating if vision embeddings are loaded
    
    // Text embedding storage for GPU batch processing (Phase 2.2)
    private java.util.Map<Integer, FloatArray> textEmbeddings; // Text token embeddings at specific positions
    
    // Vision embedding dimensions
    private final int visionEmbeddingDim;            // Dimension of vision embeddings (typically 4096)
    
    public VLMState(Configuration config, int batchsize) {
        super(config, batchsize);
        
        // Initialize vision-specific parameters
        this.visionEmbeddingDim = config.dim(); // Use model's embedding dimension
        this.hasVisionEmbeddings = false;
        this.numVisionTokens = 0;
        this.textStartPosition = 0;
        
        // Initialize text embedding storage for GPU batch processing
        this.textEmbeddings = new java.util.HashMap<>();
    }
    
    /**
     * Store vision embeddings from the vision encoder and projector.
     * This method enables direct embedding injection bypassing tokenization.
     * 
     * @param embeddings Array of vision token embeddings [numTokens x embeddingDim]
     * @param startPosition Position in sequence where vision tokens begin (typically 0)
     */
    public void setVisionEmbeddings(FloatArray[] embeddings, int startPosition) {
        this.visionEmbeddings = embeddings;
        this.numVisionTokens = embeddings.length;
        this.hasVisionEmbeddings = true;
        
        // Create position mapping for vision tokens
        this.visionTokenPositions = new int[numVisionTokens];
        for (int i = 0; i < numVisionTokens; i++) {
            this.visionTokenPositions[i] = startPosition + i;
        }
        
        // Text starts after vision tokens
        this.textStartPosition = startPosition + numVisionTokens;
    }
    
    /**
     * Store vision embeddings from a single combined FloatArray.
     * Converts the combined array into individual token embeddings.
     * 
     * @param combinedEmbeddings Combined vision embeddings [numTokens * embeddingDim]
     * @param numTokens Number of vision tokens
     * @param startPosition Position in sequence where vision tokens begin
     */
    public void setVisionEmbeddings(FloatArray combinedEmbeddings, int numTokens, int startPosition) {
        this.numVisionTokens = numTokens;
        this.visionEmbeddings = new FloatArray[numTokens];
        
        // Split combined embeddings into individual token embeddings
        for (int i = 0; i < numTokens; i++) {
            this.visionEmbeddings[i] = new FloatArray(visionEmbeddingDim);
            for (int j = 0; j < visionEmbeddingDim; j++) {
                this.visionEmbeddings[i].set(j, combinedEmbeddings.get(i * visionEmbeddingDim + j));
            }
        }
        
        this.hasVisionEmbeddings = true;
        
        // Create position mapping
        this.visionTokenPositions = new int[numTokens];
        for (int i = 0; i < numTokens; i++) {
            this.visionTokenPositions[i] = startPosition + i;
        }
        
        this.textStartPosition = startPosition + numTokens;
    }
    
    /**
     * Check if a given position corresponds to a vision token.
     * 
     * @param position Sequence position to check
     * @return true if position contains a vision token, false otherwise
     */
    public boolean isVisionPosition(int position) {
        if (!hasVisionEmbeddings) return false;
        
        for (int visionPos : visionTokenPositions) {
            if (visionPos == position) return true;
        }
        return false;
    }
    
    /**
     * Get the vision token index for a given sequence position.
     * 
     * @param position Sequence position
     * @return Vision token index, or -1 if position is not a vision token
     */
    public int getVisionTokenIndex(int position) {
        if (!hasVisionEmbeddings) return -1;
        
        for (int i = 0; i < visionTokenPositions.length; i++) {
            if (visionTokenPositions[i] == position) return i;
        }
        return -1;
    }
    
    /**
     * Get the vision embedding for a specific vision token index.
     * 
     * @param visionIndex Index of the vision token (0 to numVisionTokens-1)
     * @return Vision embedding FloatArray
     */
    public FloatArray getVisionEmbedding(int visionIndex) {
        if (!hasVisionEmbeddings || visionIndex < 0 || visionIndex >= numVisionTokens) {
            return null;
        }
        return visionEmbeddings[visionIndex];
    }
    
    /**
     * Get embedding at a specific sequence position, handling both vision and text tokens.
     * This is the key method that enables unified access to mixed embedding sources.
     * 
     * @param position Sequence position
     * @return FloatArray containing the embedding at that position
     */
    public FloatArray getEmbeddingAtPosition(int position) {
        // Check if this is a vision position
        int visionIndex = getVisionTokenIndex(position);
        if (visionIndex >= 0) {
            return getVisionEmbedding(visionIndex);
        }
        
        // Check if this is a text position with pre-embedded text token
        if (textEmbeddings.containsKey(position)) {
            return textEmbeddings.get(position);
        }
        
        // No embedding found at this position
        return null;
    }
    
    /**
     * Store a text token embedding at a specific sequence position.
     * This enables GPU batch processing for text tokens (Phase 2.2).
     * 
     * @param position Sequence position where the embedding should be stored
     * @param embedding The text token embedding to store
     */
    public void setEmbeddingAtPosition(int position, FloatArray embedding) {
        textEmbeddings.put(position, embedding);
    }
    
    /**
     * Check if there is a stored text embedding at the given position.
     * 
     * @param position Sequence position to check
     * @return true if there is a text embedding at this position
     */
    public boolean hasTextEmbeddingAtPosition(int position) {
        return textEmbeddings.containsKey(position);
    }
    
    /**
     * Get the number of stored text embeddings.
     * 
     * @return Number of text embeddings stored
     */
    public int getNumTextEmbeddings() {
        return textEmbeddings.size();
    }
    
    /**
     * Get the number of vision tokens stored in this state.
     */
    public int getNumVisionTokens() {
        return numVisionTokens;
    }
    
    /**
     * Get the position where text tokens start (after vision tokens).
     */
    public int getTextStartPosition() {
        return textStartPosition;
    }
    
    /**
     * Check if vision embeddings are loaded.
     */
    public boolean hasVisionEmbeddings() {
        return hasVisionEmbeddings;
    }
    
    /**
     * Get all vision token positions.
     */
    public int[] getVisionTokenPositions() {
        return visionTokenPositions != null ? visionTokenPositions.clone() : new int[0];
    }
    
    /**
     * Clear vision embeddings and reset vision-related state.
     */
    public void clearVisionEmbeddings() {
        this.visionEmbeddings = null;
        this.visionTokenPositions = null;
        this.hasVisionEmbeddings = false;
        this.numVisionTokens = 0;
        this.textStartPosition = 0;
    }
    
    /**
     * Clear text embeddings stored for GPU batch processing.
     */
    public void clearTextEmbeddings() {
        this.textEmbeddings.clear();
    }
    
    /**
     * Clear all embeddings (both vision and text).
     */
    public void clearAllEmbeddings() {
        clearVisionEmbeddings();
        clearTextEmbeddings();
    }
    
    /**
     * Get debug information about the VLM state.
     */
    public String getVLMDebugInfo() {
        if (!hasVisionEmbeddings) {
            return "VLMState: No vision embeddings loaded";
        }
        
        return String.format("VLMState: %d vision tokens, embedding dim=%d, text starts at position %d", 
                           numVisionTokens, visionEmbeddingDim, textStartPosition);
    }
}