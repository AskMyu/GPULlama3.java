package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.state.OlmoeState;
import org.beehive.gpullama3.inference.weights.olmoe.OlmoeTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.olmoe.OlmoeConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

/**
 * OLMoE-specific TornadoVM layer planner that uses the custom forwardTornadoVMOlmoe method.
 *
 * Unlike standard models that use generic TornadoVM kernels, OLMoE requires specialized
 * handling for MoE routing, NEOX-style RoPE, and proper state management.
 */
public class OLMoETornadoVMLayerPlanner extends TornadoVMLayerPlanner<OlmoeState, OlmoeConfiguration, OlmoeTornadoWeights> {

    /**
     * Constructs a TornadoVMLayerPlanner for OLMoE models.
     *
     * @param state The OLMoE state object containing model tensors and buffers
     * @param model The OLMoE model instance containing configuration and weights
     */
    public OLMoETornadoVMLayerPlanner(OlmoeState state, Model model) {
        super(state, model);
        System.err.println("[OLMOE-PLANNER] ✅ Initialized OLMoE-specific TornadoVM planner");
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() throws Exception {
        System.err.println("[OLMOE-PLANNER] Setting up OLMoE-specific forward plan");

        // OLMoE computation is handled by InferenceCore.forwardTornadoVMOlmoe()
        // This planner just sets up basic data flow without actual layer computation
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        // Initialize state tensors
        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);

        // Create activation update task graph - handles input data transfer
        TaskGraph activationUpdate = TornadoVMSafeInitializer.createTaskGraphSafely("olmoe_activation")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("olmoe_update", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());

        // Create logits task graph - handles output data transfer
        TaskGraph logitsTask = TornadoVMSafeInitializer.createTaskGraphSafely("olmoe_logits")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, state.wrapLogits)
                .consumeFromDevice(state.wrapX)
                .task("olmoe_logits_copy", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapLogits)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logitsTask.snapshot());

        System.err.println("[OLMOE-PLANNER] ✅ OLMoE task graphs created successfully");

        // Create minimal scheduler with worker grids for the tasks
        GridScheduler scheduler = setupOlmoeGridScheduler();
        return new Tuple2<>(taskGraphs, scheduler);
    }

    /**
     * Sets up a minimal grid scheduler for OLMoE tasks.
     * Since actual computation happens in forwardTornadoVMOlmoe, this only needs
     * to handle the data transfer tasks.
     */
    private GridScheduler setupOlmoeGridScheduler() {
        GridScheduler scheduler = new GridScheduler();

        // Single worker for data transfer tasks
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // Map workers to our minimal tasks
        scheduler.addWorkerGrid("olmoe_activation.olmoe_update", singleWorker);
        scheduler.addWorkerGrid("olmoe_logits.olmoe_logits_copy", singleWorker);

        return scheduler;
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() throws Exception {
        // Use same plan for non-NVIDIA devices
        return setupTornadoForwardPlanLayered();
    }
}