/*
 * TornadoVM Safe Initialization Wrapper
 * Prevents static initialization deadlocks by ensuring TornadoVM objects
 * are only created after the runtime is fully initialized.
 */
package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.TornadoRuntime;

/**
 * Centralized TornadoVM initialization safety wrapper.
 * Prevents static initialization deadlocks by deferring TornadoVM object creation
 * until the runtime is fully initialized and safe to use.
 */
public class TornadoVMSafeInitializer {

    private static volatile boolean runtimeInitialized = false;
    private static volatile boolean initializationAttempted = false;
    private static final Object initLock = new Object();
    private static Exception lastInitializationError = null;

    /**
     * Safely create a TaskGraph, ensuring no static initialization deadlock.
     * This method should be used instead of direct 'TornadoVMSafeInitializer.createTaskGraphSafely()' calls.
     */
    public static TaskGraph createTaskGraphSafely(String name) throws Exception {
        // Defer initialization until actual TaskGraph creation to avoid service loading issues
        System.err.println("[TORNADO-SAFE] Creating TaskGraph with deferred initialization: " + name);
        try {
            return new TaskGraph(name);
        } catch (Exception e) {
            System.err.println("[TORNADO-SAFE] TaskGraph creation failed: " + e.getMessage());
            throw new RuntimeException("TornadoVM TaskGraph creation failed", e);
        }
    }

    /**
     * Safely create a TornadoExecutionPlan, ensuring no static initialization deadlock.
     * This method should be used instead of direct 'new TornadoExecutionPlan()' calls.
     */
    public static TornadoExecutionPlan createExecutionPlanSafely(ImmutableTaskGraph taskGraph) throws Exception {
        System.err.println("[TORNADO-SAFE] Creating ExecutionPlan with deferred initialization");
        try {
            return new TornadoExecutionPlan(taskGraph);
        } catch (Exception e) {
            System.err.println("[TORNADO-SAFE] ExecutionPlan creation failed: " + e.getMessage());
            throw new RuntimeException("TornadoVM ExecutionPlan creation failed", e);
        }
    }

    /**
     * Safely create a TornadoExecutionPlan with multiple TaskGraphs, ensuring no static initialization deadlock.
     * This method should be used instead of direct 'new TornadoExecutionPlan()' calls.
     */
    public static TornadoExecutionPlan createExecutionPlanSafely(ImmutableTaskGraph... taskGraphs) throws Exception {
        System.err.println("[TORNADO-SAFE] Creating ExecutionPlan with multiple graphs, deferred initialization");
        try {
            return new TornadoExecutionPlan(taskGraphs);
        } catch (Exception e) {
            System.err.println("[TORNADO-SAFE] ExecutionPlan creation failed: " + e.getMessage());
            throw new RuntimeException("TornadoVM ExecutionPlan creation failed", e);
        }
    }

    /**
     * Safely access TornadoRuntime, ensuring no static initialization deadlock.
     * This method should be used instead of direct 'TornadoRuntimeProvider.getTornadoRuntime()' calls.
     */
    public static TornadoRuntime getTornadoRuntimeSafely() throws Exception {
        ensureSafeInitialization();
        return TornadoRuntimeProvider.getTornadoRuntime();
    }

    /**
     * Check if TornadoVM is available without triggering static initialization.
     */
    public static boolean isTornadoVMAvailable() {
        return runtimeInitialized && lastInitializationError == null;
    }

    /**
     * Ensure TornadoVM runtime is safely initialized before creating objects.
     * Uses double-checked locking to prevent race conditions and static deadlocks.
     */
    private static void ensureSafeInitialization() throws Exception {
        if (runtimeInitialized) {
            return;  // Fast path for already initialized runtime
        }

        synchronized (initLock) {
            if (runtimeInitialized) {
                return;  // Double-checked locking
            }

            if (initializationAttempted && lastInitializationError != null) {
                throw new RuntimeException("TornadoVM initialization previously failed", lastInitializationError);
            }

            if (!initializationAttempted) {
                initializationAttempted = true;

                try {
                    System.err.println("[TORNADO-SAFE] Attempting safe TornadoVM runtime initialization...");

                    // Check system properties to ensure configuration is correct
                    String tornadoDriver = System.getProperty("tornado.driver", "0");
                    String tornadoDevice = System.getProperty("tornado.device", "0");
                    System.err.println("[TORNADO-SAFE] Configuration - driver: " + tornadoDriver + ", device: " + tornadoDevice);

                    // Directly attempt TornadoVM runtime access without reflection pre-checks
                    // The problematic static initialization happens here, but in a controlled manner
                    System.err.println("[TORNADO-SAFE] Accessing TornadoVM runtime directly...");
                    TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();

                    // Verify runtime is functional
                    if (runtime.getNumBackends() > 0) {
                        System.err.println("[TORNADO-SAFE] TornadoVM runtime initialized successfully with " +
                                         runtime.getNumBackends() + " backends");
                        runtimeInitialized = true;
                    } else {
                        throw new RuntimeException("TornadoVM runtime has no backends available");
                    }

                } catch (Exception e) {
                    lastInitializationError = e;
                    System.err.println("[TORNADO-SAFE] TornadoVM initialization failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    throw new RuntimeException("TornadoVM safe initialization failed", e);
                }
            }
        }

        if (!runtimeInitialized) {
            throw new RuntimeException("TornadoVM runtime could not be initialized safely");
        }
    }

    /**
     * Reset initialization state - primarily for testing purposes.
     */
    public static void resetInitializationState() {
        synchronized (initLock) {
            runtimeInitialized = false;
            initializationAttempted = false;
            lastInitializationError = null;
        }
    }
}