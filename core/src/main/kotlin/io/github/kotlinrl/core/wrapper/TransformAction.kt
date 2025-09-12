package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

/**
 * A wrapper class that transforms the actions of a wrapped environment
 * before they are executed. This enables a custom mapping from abstract
 * actions to the specific actions understood by the wrapped environment.
 *
 * The `TransformAction` class modifies only the action space and behavior
 * of the `step` function while keeping the observation space and other
 * properties of the wrapped environment unchanged.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the abstract action before transformation.
 * @param ObservationSpace The type representing the space of observations.
 * @param ActionSpace The type representing the space of abstract actions.
 * @param WrappedAction The type of action used in the wrapped environment.
 * @param WrappedActionSpace The type representing the space of wrapped actions.
 * @param env The wrapped environment whose actions need to be transformed.
 * @param transform A function that maps the abstract action to the wrapped action.
 * @param actionSpace The space of abstract actions in the transformed environment.
 */
class TransformAction<State, Action,
        ObservationSpace : Space<State>,
        ActionSpace : Space<Action>,
        WrappedAction,
        WrappedActionSpace : Space<WrappedAction>>(
    env: Env<State, WrappedAction, ObservationSpace, WrappedActionSpace>,
    private val transform: (Action) -> WrappedAction,
    override val actionSpace: ActionSpace
) : Wrapper<State, Action, ObservationSpace, ActionSpace, State, WrappedAction, ObservationSpace, WrappedActionSpace>(env) {

    /**
     * Exposes the observation space of the wrapped environment.
     *
     * This represents the space from which observations are drawn for the transformed environment.
     * The observation space remains identical to that of the underlying wrapped environment.
     */
    override val observationSpace: ObservationSpace
        get() = env.observationSpace

    /**
     * Resets the wrapped environment to its initial state.
     *
     * This method delegates the reset call to the wrapped environment, reinitializing it
     * and setting up a new episode. It optionally utilizes a random seed and additional
     * options for customization.
     *
     * @param seed An optional random seed to enable deterministic resetting behavior.
     *             If `null`, the wrapped environment's default random generator is used.
     * @param options An optional map of configuration options to influence the reset behavior
     *                in the wrapped environment. The keys and values depend on the specific
     *                implementation of the wrapped environment.
     * @return The initial state of the environment post-reset, wrapped in an `InitialState`
     *         object. This includes the reset state and any relevant metadata provided in
     *         the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> =
        env.reset(seed, options)

    /**
     * Executes a single step in the environment, applying a transformation to the given action
     * before delegating the step execution to the wrapped environment.
     *
     * @param action The action to be transformed and executed in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and additional information after the transformed action is applied.
     */
    override fun step(action: Action): StepResult<State> {
        val mapped = transform(action)
        return env.step(mapped)
    }
}
