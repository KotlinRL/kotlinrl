package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

/**
 * A wrapper class for environments that provides direct delegation of the core functionalities
 * to the wrapped environment. It simplifies the creation of custom environment wrappers
 * by extending from this class and overriding specific methods as needed.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing the actions that can be taken in the environment.
 * @param ObservationSpace The type of the observation space for the given state.
 * @param ActionSpace The type of the action space for the given actions.
 * @constructor Initializes the wrapper with the specified environment.
 * @param env The environment to wrap and delegate operations to.
 */
open class SimpleWrapper<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>
) : Wrapper<State, Action, ObservationSpace, ActionSpace, State, Action, ObservationSpace, ActionSpace>(env) {

    /**
     * Executes a single step in the environment using the given action.
     * The method delegates the step execution to the wrapped environment and
     * returns the resulting state, reward, termination status, truncation status,
     * and auxiliary information.
     *
     * @param action The action to perform in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and additional information after performing the action.
     */
    override fun step(action: Action): StepResult<State> = env.step(action)

    /**
     * Resets the wrapped environment to an initial state.
     *
     * This method delegates the reset operation to the underlying environment,
     * optionally using the provided random seed and configuration options.
     *
     * @param seed An optional random seed to ensure deterministic behavior. If `null`, the default random generator is used.
     * @param options An optional map of configuration options that may influence the reset process.
     * @return The initial state of the environment, encapsulated in an `InitialState`, which includes the state and metadata.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> = env.reset(seed, options)

    /**
     * Represents the observation space of the wrapped environment.
     *
     * This property retrieves the observation space directly from the underlying
     * environment, providing information about the structure and boundaries
     * of observations that the environment can produce.
     */
    override val observationSpace: ObservationSpace
        get() = env.observationSpace

    /**
     * Represents the action space of the environment this wrapper is interacting with.
     *
     * This property provides an abstraction of all possible actions that can be taken in the underlying environment.
     * It is used to query and validate the space of legal actions for the environment.
     *
     * Delegates to the `actionSpace` property of the wrapped environment.
     */
    override val actionSpace: ActionSpace
        get() = env.actionSpace
}