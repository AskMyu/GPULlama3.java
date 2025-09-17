package org.beehive.gpullama3.process;

import java.io.*;

/**
 * Utilities for serializing data for inter-process communication.
 * Optimized for GPU operation data that needs to be passed between processes.
 */
public class SerializationUtils {
    
    /**
     * Serialize an object to a file.
     */
    public static void serialize(Serializable obj, File file) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(file);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(obj);
            oos.flush();
        }
    }
    
    /**
     * Deserialize an object from a file.
     */
    @SuppressWarnings("unchecked")
    public static <T> T deserialize(File file, Class<T> expectedClass) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(file);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            Object obj = ois.readObject();
            
            if (!expectedClass.isInstance(obj)) {
                throw new ClassCastException("Expected " + expectedClass.getName() + 
                    " but got " + (obj != null ? obj.getClass().getName() : "null"));
            }
            
            return expectedClass.cast(obj);
        }
    }
    
    /**
     * Serialize to byte array for in-memory operations.
     */
    public static byte[] serializeToBytes(Serializable obj) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(obj);
            oos.flush();
            return baos.toByteArray();
        }
    }
    
    /**
     * Deserialize from byte array.
     */
    @SuppressWarnings("unchecked")
    public static <T> T deserializeFromBytes(byte[] data, Class<T> expectedClass) 
            throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bais = new ByteArrayInputStream(data);
             ObjectInputStream ois = new ObjectInputStream(bais)) {
            Object obj = ois.readObject();
            
            if (!expectedClass.isInstance(obj)) {
                throw new ClassCastException("Expected " + expectedClass.getName() + 
                    " but got " + (obj != null ? obj.getClass().getName() : "null"));
            }
            
            return expectedClass.cast(obj);
        }
    }
    
    /**
     * Check if an object is serializable.
     */
    public static boolean isSerializable(Object obj) {
        return obj instanceof Serializable;
    }
    
    /**
     * Get the serialized size of an object in bytes.
     */
    public static long getSerializedSize(Serializable obj) {
        try {
            byte[] data = serializeToBytes(obj);
            return data.length;
        } catch (IOException e) {
            return -1;
        }
    }
    
    /**
     * Validate that an object can be serialized and deserialized successfully.
     */
    public static boolean validateSerialization(Serializable obj, Class<?> expectedClass) {
        try {
            byte[] data = serializeToBytes(obj);
            Object deserialized = deserializeFromBytes(data, expectedClass);
            return deserialized != null && expectedClass.isInstance(deserialized);
        } catch (Exception e) {
            return false;
        }
    }
}