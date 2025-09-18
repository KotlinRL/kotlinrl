package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*

/**
 * A wrapper for environments that enforces a time limit on episodes.
 * The episode terminates if the elapsed steps exceed the specified maximum number of steps,
 * even if the underlying environment has not naturally terminated.
 *
 * This class is useful for controlling the length of an episode in reinforcement learning environments
 * and ensuring agents do not interact with the environment indefinitely.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space defining the structure of observation states.
 * @param ActionSpace The type of space defining the structure of allowable actions.
 * @param env The environment to wrap and enforce the time limit on.
 * @param maxEpisodeSteps The maximum number of steps allowed per episode. Once reached,
 *                        the episode is truncated if the environment has not already terminated.
 */
class TimeLimit<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val maxEpisodeSteps: Int
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var elapsedSteps = 0

    /**
     * Resets the environment to an initial state and reinitializes internal counters.
     *
     * This method resets the environment to start a new episode. It optionally accepts
     * a random seed and configuration options to customize the reset behavior.
     *
     * @param seed An optional random seed for deterministic behavior. If `null`, no specific seed is used.
     * @param options An optional map of configuration options to influence the reset process.
     *                The specific keys and values depend on the environment implementation.
     * @return The initial state of the environment after reset, encapsulated in an `InitialState`.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        elapsedSteps = 0
        return env.reset(seed, options)
    }

    /**
     * Executes a single step in the environment with a time limit constraint, based on the provided action.
     * This method wraps around the environment's step function to monitor and enforce episode length limits.
     * If the time limit is reached, additional metadata is added to indicate truncation due to the limit.
     *
     * @param action The action to be performed in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status (potentially updated to indicate time limit truncation), and additional metadata.
     */
    override fun step(action: Action): StepResult<State> {
        val transition = env.step(action)
        elapsedSteps += 1

        val reachedTimeLimit = elapsedSteps >= maxEpisodeSteps

        val newTruncated = transition.truncated || (reachedTimeLimit && !transition.terminated)
        val newTerminated = transition.terminated

        val newInfo = transition.info.toMutableMap()
        if (reachedTimeLimit && !transition.terminated) {
            newInfo["TimeLimit.truncated"] = "true"
        }

        return transition.copy(
            terminated = newTerminated,
            truncated = newTruncated,
            info = newInfo
        )
    }
}
