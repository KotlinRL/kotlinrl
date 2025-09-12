package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

/**
 * A wrapper class that applies a transformation to the state of an environment.
 *
 * This class wraps around an existing environment and transforms its raw states
 * into a new representation defined by the provided transformation function.
 * It maintains the action and observation spaces of the wrapped environment,
 * while exposing transformed states to the user.
 *
 * @param State The type of the transformed state output by this wrapper.
 * @param Action The type of actions supported by the environment.
 * @param ObservationSpace The observation space type associated with the transformed states.
 * @param ActionSpace The action space type associated with the wrapped environment.
 * @param WrappedState The raw state type of the wrapped environment before transformation.
 * @param WrappedObservationSpace The observation space type associated with the raw states.
 * @param env The environment being wrapped, responsible for managing the core logic
 *            of state transitions and interactions with actions.
 * @param transform A transformation function that converts the raw state of the
 *                  wrapped environment into the desired state representation used
 *                  in this wrapper.
 * @param observationSpace The observation space corresponding to the transformed states.
 */
class TransformState<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>, WrappedState, WrappedObservationSpace : Space<WrappedState>>(
    env: Env<WrappedState, Action, WrappedObservationSpace, ActionSpace>,
    private val transform: (WrappedState) -> State,
    override val observationSpace: ObservationSpace
) : Wrapper<State, Action, ObservationSpace, ActionSpace, WrappedState, Action, WrappedObservationSpace, ActionSpace>(
    env
) {

    /**
     * Resets the environment and transforms the initial state.
     *
     * This method initializes the environment by calling the `reset` method on the
     * underlying environment, optionally providing a random seed and additional options.
     * After resetting, it transforms the initial state and returns the result.
     *
     * @param seed An optional random seed used to initialize the environment. If `null`,
     *             the default random generator of the environment is used.
     * @param options A map of optional configurations that can modify the behavior
     *                of the reset process. Specific details depend on the implementation.
     * @return An `InitialState` instance encapsulating the transformed initial state
     *         and additional metadata information.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        val initial = env.reset(seed, options)
        return InitialState(state = transform(initial.state), info = initial.info)
    }

    /**
     * Executes a single step in the environment using the provided action and returns the result
     * with the state transformed as per the current transformation logic.
     *
     * @param action The action to be applied in the environment.
     * @return A StepResult containing the transformed state, reward obtained,
     * termination status, truncation status, and additional metadata after the action is executed.
     */
    override fun step(action: Action): StepResult<State> {
        val transition = env.step(action)
        return StepResult(
            state = transform(transition.state),
            reward = transition.reward,
            terminated = transition.terminated,
            truncated = transition.truncated,
            info = transition.info
        )
    }

    /**
     * Represents the action space of the transformed environment.
     *
     * This property provides access to the underlying environment's action space, allowing
     * users to retrieve the available actions that can be taken in the current environment state.
     * It is derived from the `actionSpace` of the encapsulated environment, ensuring that
     * transformations applied by the containing class do not alter the definition of valid actions.
     */
    override val actionSpace: ActionSpace
        get() = env.actionSpace
}

