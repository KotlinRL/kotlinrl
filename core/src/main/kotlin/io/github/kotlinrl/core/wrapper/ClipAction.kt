package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * A wrapper class for modifying actions in an environment by clipping them to a constrained range
 * defined by the action space of the environment. The class is generic in the state, numeric type,
 * dimension, and observation space of the environment.
 *
 * @param State The type representing the state of the environment.
 * @param Num The numeric type used for the actions and observations within the environment.
 * @param D The dimension type for the environment's action and observation spaces.
 * @param ObservationSpace The specific observation space associated with the environment.
 * @param env The environment instance to be wrapped and managed. It defines action and observation spaces
 * and handles state transitions based on incoming actions.
 */
class ClipAction<
        State,
        Num : Number,
        D : Dimension,
        ObservationSpace
        : Space<State>
        >(
    env: Env<State, NDArray<Num, D>, ObservationSpace, Box<Num, D>>
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
     * The action space of the environment, represented as a bounded space (`Box`) that defines the
     * permissible range for each action value. This property provides access to the `Box` instance
     * of the wrapped environment, ensuring consistency in action constraints.
     *
     * The `Box` defines the dimensionality and numeric data type of the space, as well as the
     * lower and upper bounds for valid actions. These bounds are enforced to ensure
     * that actions remain within the predefined constraints.
     *
     * @see Box
     */
    override val actionSpace: Box<Num, D>
        get() = env.actionSpace

    /**
     * Represents the observation space of the environment being wrapped.
     * This property provides access to the structure and characteristics of the observations
     * that the environment can generate, such as shape, bounds, and data type.
     *
     * The observation space is defined by the underlying environment and is exposed here
     * to allow interaction and consistency with the wrapped environment.
     */
    override val observationSpace: ObservationSpace
        get() = env.observationSpace

    /**
     * Resets the environment to its initial state with optional customization.
     *
     * This method reinitializes the environment, allowing it to start a new episode.
     * Optionally, a random seed and a set of configuration options can be provided
     * to control the reset behavior.
     *
     * @param seed An optional random seed for deterministic environment behavior. If `null`, the default random generator is used.
     * @param options An optional set of configuration options to influence the reset process, specific to the environment implementation.
     * @return The initial state of the environment, encapsulated in an `InitialState` object, which includes the state and any auxiliary metadata in the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> =
        env.reset(seed, options)

    /**
     * Executes a single step in the environment with the provided action after applying clipping
     * to ensure the action is within the environment's defined action space.
     *
     * @param action The action to be performed, represented as an NDArray.
     *               It is clipped to ensure compliance with the action space of the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     *         truncation status, and any auxiliary information after the action is applied.
     */
    override fun step(action: NDArray<Num, D>): StepResult<State> {
        val clipped = clipToBox(action, env.actionSpace)
        return env.step(clipped)
    }
}
