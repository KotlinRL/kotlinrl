package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*

/**
 * A wrapper for normalizing rewards in an environment to ensure they are standardized
 * based on mean and standard deviation, which makes the reward distribution more stable
 * and suitable for training reinforcement learning algorithms.
 *
 * This class uses `RunningStats` to track the mean and standard deviation of rewards observed
 * over time. It modifies the rewards by subtracting the mean and dividing by the standard deviation.
 * A small constant `epsilon` is added to the denominator to prevent division by zero.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the observation space for the state.
 * @param ActionSpace The type of space specifying the allowable actions in the environment.
 * @param env The environment on which this wrapper operates.
 * @param epsilon A small constant added to prevent division by zero during normalization.
 */
class NormalizeReward<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val epsilon: Double = 1e-8
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private val stats = RunningStats()

    /**
     * Resets the wrapped environment to its initial state.
     *
     * This method allows reinitialization of the environment, starting a new episode
     * while passing an optional random seed and additional options to customize
     * the reset behavior. The reset operation delegates to the underlying environment.
     *
     * @param seed An optional random seed for producing deterministic behavior. If `null`,
     *             the default random generator of the environment will be used.
     * @param options An optional map of configuration options to adjust the reset process.
     *                The specific options depend on the underlying environment's implementation.
     * @return The initial state of the environment after the reset operation, encapsulated
     *         in an `InitialState`, including the state and metadata within the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> =
        env.reset(seed, options)

    /**
     * Performs a single step in the environment with reward normalization.
     *
     * This method executes the given action in the environment, retrieves the resulting step data,
     * and normalizes the reward using the statistics tracked in `stats`. The normalization is
     * computed by subtracting the mean reward from the current reward and dividing by the
     * standard deviation, with a minimum threshold defined by `epsilon` to avoid division by zero.
     * The normalized reward is then returned as part of the step result.
     *
     * @param action The action to execute within the environment.
     * @return A StepResult containing the updated state, normalized reward, termination status,
     * truncation status, and additional information after executing the action.
     */
    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        stats.update(t.reward)
        val normalized = (t.reward - stats.mean) / maxOf(stats.std, epsilon)
        return t.copy(reward = normalized)
    }
}