package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

/**
 * Enforces proper order of operations within an environment by ensuring that `reset`
 * is called before any `step` operations. This wrapper will throw an exception if
 * `step` is called either before the environment is reset or after an episode has
 * terminated or been truncated.
 *
 * This class wraps an existing environment, adding stricter control over its usage
 * to avoid misuse or invalid operation sequences.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the structure of observations.
 * @param ActionSpace The type of space specifying the structure of actions.
 * @property env The environment to be wrapped by this class.
 */
class OrderEnforcing<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var needsReset = true

    /**
     * Resets the environment to an initial state and updates the reset requirement flag.
     *
     * This method overrides the base `reset` function to ensure proper operational order
     * in the environment. It reinitializes the environment for a new episode, optionally
     * taking a seed for reproducibility and additional options for customization.
     *
     * @param seed An optional random seed to produce deterministic behavior. If `null`, the default seed is used.
     * @param options An optional map of configuration options to influence the reset process.
     *                The specific keys and semantics depend on the environment implementation.
     * @return The initial state of the environment encapsulated in an `InitialState` object,
     *         which includes the state and associated metadata in the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        needsReset = false
        return env.reset(seed, options)
    }

    /**
     * Executes a single step in the environment based on the provided action, ensuring
     * that the operational order of the environment is respected. If the environment
     * requires a reset prior to this step call, an `IllegalStateException` is thrown.
     *
     * Updates the reset requirement flag if the episode has terminated or been truncated
     * after performing the action.
     *
     * @param action The action to be performed in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and additional information after the action is executed.
     * @throws IllegalStateException if the step is called before a reset after the episode is done.
     */
    override fun step(action: Action): StepResult<State> {
        if (needsReset) {
            throw IllegalStateException(
                "step() called beforeStep reset(), or afterStep episode done. " +
                        "You must call reset() beforeStep step()."
            )
        }
        val t = env.step(action)
        if (t.terminated || t.truncated) {
            needsReset = true
        }
        return t
    }
}
