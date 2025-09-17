package org.beehive.gpullama3.isolation.core;

import java.io.*;

/**
 * Generic interface for serializing objects for inter-process communication.
 */
public interface ProcessSerializer<T> {
    
    /**
     * Serialize an object to an output stream for subprocess communication.
     */
    void serialize(T object, OutputStream output) throws SerializationException;
    
    /**
     * Deserialize an object from an input stream from subprocess communication.
     */
    T deserialize(InputStream input) throws SerializationException;
    
    /**
     * Get the serialization format identifier for debugging.
     */
    String getSerializationFormat();
    
    /**
     * Exception thrown during serialization/deserialization operations.
     */
    class SerializationException extends Exception {
        public SerializationException(String message) {
            super(message);
        }
        
        public SerializationException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}