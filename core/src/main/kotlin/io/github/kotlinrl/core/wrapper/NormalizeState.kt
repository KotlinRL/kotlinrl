package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * A wrapper class that normalizes the observations of an environment based on specified
 * mean and standard deviation values. This is particularly useful in environments where
 * observations need to be scaled or standardized for consistent processing by agents.
 *
 * @param Num The numeric type of the observations, extending `Number`.
 * @param D The dimensionality of the `NDArray`.
 * @param Action The type of actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the structure and characteristics
 *                         of the observation space.
 * @param ActionSpace The type of space specifying the allowable actions in the environment.
 * @param env The underlying wrapped environment whose observations will be normalized.
 * @param mean An `NDArray` representing the mean values for normalization, conforming
 *             to the shape of the observations.
 * @param std An `NDArray` representing the standard deviation values for normalization,
 *            conforming to the shape of the observations.
 * @param epsilon A small constant added to avoid division by zero during normalization.
 */
class NormalizeState<Num : Number, D : Dimension, Action, ObservationSpace : Space<NDArray<Num, D>>, ActionSpace : Space<Action>>(
    env: Env<NDArray<Num, D>, Action, ObservationSpace, ActionSpace>,
    private val mean: NDArray<Num, D>,
    private val std: NDArray<Num, D>,
    private val epsilon: Double = 1e-8
) : SimpleWrapper<NDArray<Num, D>, Action, ObservationSpace, ActionSpace>(env) {

    private fun normalize(obs: NDArray<Num, D>): NDArray<Num, D> {
        val obsArr = obs.data
        val meanArr = mean.data
        val stdArr = std.data
        val normed = DoubleArray(obsArr.size) { i ->
            (obsArr[i].toDouble() - meanArr[i].toDouble()) / maxOf(stdArr[i].toDouble(), epsilon)
        }
        @Suppress("UNCHECKED_CAST")
        return mk.ndarray(normed.toList(), obs.shape, obs.dim) as NDArray<Num, D>
    }

    /**
     * Resets the environment to an initial normalized state.
     *
     * This method reinitializes the environment using a potential random seed and custom
     * options, then normalizes the initial state before returning it. The normalization
     * ensures that the state values are scaled based on pre-defined statistics (mean, std, epsilon).
     *
     * @param seed An optional random seed for reproducing deterministic behavior.
     *             If `null`, the environment will use its default random generator.
     * @param options An optional map of configuration parameters that influence the reset behavior,
     *                specific to the environment implementation.
     * @return The normalized initial state of the environment after the reset, encapsulated in an `InitialState`,
     *         including the normalized state and any associated metadata in the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<NDArray<Num, D>> {
        val initial = env.reset(seed, options)
        return InitialState(state = normalize(initial.state), info = initial.info)
    }

    /**
     * Executes a single step in the environment, performs the specified action, and normalizes
     * the resulting state of the environment.
     *
     * The method interacts with the environment to execute the given action, retrieves the result
     * of the step, and applies normalization to the resulting state using pre-defined statistics
     * (mean, standard deviation, and epsilon). The normalized state and other step information
     * (reward, termination, truncation, and metadata) are returned.
     *
     * @param action The action to be performed in the environment. This represents the agent's decision for the current step.
     * @return A StepResult containing the normalized updated state, reward obtained, flags for termination and truncation,
     * and additional metadata after the action is applied.
     */
    override fun step(action: Action): StepResult<NDArray<Num, D>> {
        val t = env.step(action)
        return t.copy(state = normalize(t.state))
    }
}
