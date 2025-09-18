package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * A wrapper that rescales the action space of an environment.
 *
 * The RescaleAction class modifies the action space of a given environment by rescaling
 * the range of actions from a specified minimum and maximum range to match the inner
 * action space of the wrapped environment. This is useful, for instance, when the
 * policy or agent operates within a normalized range (e.g., [-1, 1] or [0, 1])
 * while the environment accepts a different range of actions.
 *
 * @param State The type representing the state of the environment.
 * @param Num A numeric type representing the elements of the action space.
 * @param D The dimensionality of the action space.
 * @param ObservationSpace A type extending Space<State>, representing the observation space of the environment to be wrapped.
 * @param env The underlying environment to wrap. Rescaling is applied to its action space.
 * @param minAction The minimum bound of the rescaled action space. Typically filled with values like -1 or 0.
 * @param maxAction The maximum bound of the rescaled action space. Typically filled with values like 1.
 */
class RescaleAction<State, Num : Number, D : Dimension, ObservationSpace : Space<State>>(
    env: Env<State, NDArray<Num, D>, ObservationSpace, Box<Num, D>>,
    private val minAction: NDArray<Num, D>, // typically filled with -1 or 0
    private val maxAction: NDArray<Num, D>  // typically filled with 1
) : Wrapper<
        State,
        NDArray<Num, D>,
        ObservationSpace,
        Box<Num, D>,
        State,
        NDArray<Num, D>,
        ObservationSpace,
        Box<Num, D>
        >(env) {

    /**
     * Represents the action space as a bounded multidimensional numerical space, exposed to the
     * policy agent. The action space is defined by the lower and upper bounds (`minAction` and `maxAction`)
     * for each dimension, along with the data type of the numerical values.
     *
     * The `actionSpace` determines the range of permissible actions that the agent can take
     * at any given step, based on the defined constraints. The underlying implementation uses
     * the `Box` class, which enforces validation of actions to ensure they lie within the
     * specified bounds.
     *
     * This property is primarily used to interact with the policy agent for action generation
     * within the environment.
     */
// The policyAgent-facing action space
    override val actionSpace: Box<Num, D> = Box(minAction, maxAction, minAction.dtype)

    /**
     * The `observationSpace` property provides access to the observation space of the environment
     * associated with this class. The observation space defines the structure, type, and bounds
     * of the observations that the environment produces. This is often used in reinforcement
     * learning to understand the shape and constraints of input data.
     *
     * This property is overridden from the parent class and directly delegates to the
     * observation space of the encapsulated environment.
     */
    override val observationSpace: ObservationSpace
        get() = env.observationSpace

    /**
     * Resets the environment to an initial state with potential rescaling of actions.
     *
     * This method reinitializes the environment, potentially applying any rescaling or adjustments
     * defined in the `RescaleAction` class. It accepts an optional seed for reproducibility
     * and a map of additional options for custom reset behavior.
     *
     * @param seed An optional random seed for deterministic resets. If null, the default random generator is used.
     * @param options An optional map containing additional configuration options for the reset process.
     * @return The initial state of the environment encapsulated in an `InitialState`, including the environment state
     *         and any associated metadata in the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> =
        env.reset(seed, options)

    /**
     * Executes a single step in the environment with the provided action, applying rescaling
     * to match the environment's action space. The rescaling adjusts the input action's range
     * to align with the environment's expected action boundaries before executing the step.
     *
     * @param action The action to be performed, represented as an NDArray of numeric values within a specified dimension.
     * @return A StepResult containing the updated environment state, the reward obtained,
     * termination status, truncation status, and any additional metadata or auxiliary information.
     */
    override fun step(action: NDArray<Num, D>): StepResult<State> {
        val innerBox = env.actionSpace
        val scaled = rescale(
            x = action,
            srcLow = minAction,
            srcHigh = maxAction,
            tgtLow = innerBox.low,
            tgtHigh = innerBox.high,
            dim = action.dim
        )
        return env.step(scaled)
    }
}
