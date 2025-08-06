package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionEstimator

/**
 * A Q-function estimator using the Incremental Monte Carlo method.
 *
 * This class provides an implementation of the Incremental Monte Carlo algorithm
 * for estimating Q-values based on trajectories. The algorithm processes a given
 * trajectory of state-action-reward transitions to incrementally update the Q-function
 * with each visit to a state-action pair. The Q-value updates are controlled by a
 * learning rate (alpha) and the discount factor (gamma).
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the possible actions within the environment.
 * @param gamma the discount factor in the range [0, 1], determining the importance of future rewards.
 * @param alpha a parameter schedule determining the learning rate for updates.
 *              Defaults to a constant value of 0.05.
 * @param firstVisitOnly if true, the algorithm only updates the Q-value for the first visit
 *                       of each state-action pair in a trajectory. If false, every visit is considered
 *                       in the Q-value update process.
 */
class IncrementalMonteCarloQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    private val firstVisitOnly: Boolean = true,
) : TrajectoryQFunctionEstimator<State, Action> {

    /**
     * Estimates a new Q-function using an incremental Monte Carlo method based on a given trajectory
     * and an existing Q-function. The method iteratively computes return values (G) for each step
     * in the trajectory and updates the Q-values for state-action pairs based on the first-visit or
     * every-visit Monte Carlo approach, as determined by the configuration.
     *
     * @param Q the current Q-function, mapping state-action pairs to their quality values.
     * @param trajectory the trajectory consisting of a sequence of state, action, and reward tuples,
     *                   used for updating the Q-function.
     * @return an updated Q-function reflecting the adjustments made using the given trajectory.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        var currentQ = Q
        val visited = mutableSetOf<StateActionKey<*, *>>()
        var G = 0.0

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = StateActionKey(s, a)

            if (firstVisitOnly && key in visited) continue
            visited.add(key)

            val oldQ = currentQ[s, a]
            val updatedQ = oldQ + alpha() * (G - oldQ)
            currentQ = currentQ.update(s, a, updatedQ)
        }

        return currentQ
    }
}
