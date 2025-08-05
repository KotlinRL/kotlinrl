package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.space.*
import kotlin.random.*

/**
 * Represents a generic environment interface that facilitates interaction between
 * agents and their surroundings. The environment provides mechanisms for stepping
 * through actions, resetting its state, rendering its current state, and managing
 * spaces for observations and actions.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the observation space for the state.
 * @param ActionSpace The type of space specifying the allowable actions in the environment.
 */
interface Env<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>> {
    /**
     * Executes a single step in the environment based on the provided action.
     * The environment updates its state and returns the resulting state, reward,
     * termination status, truncation status, and auxiliary information.
     *
     * @param action The action to be performed in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and additional information after the action is applied.
     */
    fun step(action: Action): StepResult<State>
    /**
     * Resets the environment to an initial state.
     *
     * This method reinitializes the environment, allowing it to start a new episode.
     * It optionally takes a random seed and additional options to customize the reset behavior.
     *
     * @param seed An optional random seed for reproducing deterministic behavior.
     *             If `null`, the environment will use its default random generator.
     * @param options An optional map of configuration options to influence the reset process.
     *                The specific keys and values depend on the environment implementation.
     * @return The initial state of the environment after reset, encapsulated in an `InitialState`.
     *         This includes the state and any associated metadata in the `info` map.
     */
    fun reset(seed: Int? = null, options: Map<String, Any?>? = null): InitialState<State>
    /**
     * Renders the current state of the environment.
     *
     * This method provides a visual or data representation of the environment's current state.
     * The returned `Rendering` object can either represent an empty state or include a frame
     * with specific dimensions and raw image data.
     *
     * @return A `Rendering` instance representing the current state of the environment.
     */
    fun render(): Rendering
    /**
     * Closes the environment and releases any allocated resources.
     *
     * This method should be called to properly clean up resources or connections associated
     * with the environment. Once called, the environment should no longer be used.
     */
    fun close()
    /**
     * Contains additional information or metadata about the environment.
     *
     * This map can hold arbitrary key-value pairs that describe properties
     * or behaviors of the environment. The specific contents and structure
     * of the metadata depend on the particular implementation of the environment.
     *
     * Potential uses of the metadata include providing human-readable descriptions,
     * specifying versioning information, or configuring environment details.
     */
    val metadata: Map<String, Any?>
    /**
     * Represents the observation space of the environment.
     *
     * The observation space defines the structure and constraints of the observations
     * that the environment can produce. Observations are returned after performing
     * actions in the environment or when it is reset. This property provides the
     * means to describe or validate the format, type, and boundaries of these observations.
     */
    val observationSpace: ObservationSpace
    /**
     * Defines the space of all possible actions that can be performed in the environment.
     *
     * The `actionSpace` represents the set of valid actions that the agent can take
     * at any given step in the environment. It is used to sample random actions,
     * validate incoming actions, or encode specific structural constraints on the actions.
     */
    val actionSpace: ActionSpace
    /**
     * Provides access to the random number generator used in the environment.
     *
     * This property allows for reproducibility and control over random processes
     * within the environment, such as sampling actions, initializing states, or
     * introducing stochasticity in transitions. The specific implementation of
     * the random generator may vary depending on the environment.
     */
    val random: Random
}
