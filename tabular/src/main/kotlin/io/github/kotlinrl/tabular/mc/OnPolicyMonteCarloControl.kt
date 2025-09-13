package io.github.kotlinrl.tabular.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.agent.Trajectory
import io.github.kotlinrl.core.algorithm.TrajectoryLearningAlgorithm
import io.github.kotlinrl.core.api.Policy
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * On-policy Monte Carlo control algorithm implementation for reinforcement learning.
 *
 * This algorithm is designed to optimize a policy using the Monte Carlo method to estimate
 * action values (Q-values) based on observed trajectories of state-action-reward interactions.
 * It updates the Q-table iteratively and adjusts the policy to become greedy with respect to
 * the updated Q-values.
 *
 * Key features of the algorithm:
 * - Supports both first-visit and every-visit Monte Carlo methods for Q-value updates.
 * - Uses a Q-table to store Q-values for state-action pairs.
 * - Allows custom policy updates via a provided `PolicyUpdate` function.
 * - Processes entire trajectories to derive cumulative rewards (returns) and update Q-values.
 *
 * @param onPolicyUpdate the function invoked to update the policy based on the learned Q-values.
 * @param rng random number generator used for probabilistic processes.
 * Defaults to `Random.Default`.
 * @param initialPolicy the starting policy used by the algorithm to select actions.
 * @param Q the Q-table for storing action-value estimates.
 * @param onQUpdate the callback function triggered after updates to the Q-table.
 * @param gamma the discount factor for future rewards, used in return calculations.
 * @param firstVisitOnly determines whether the algorithm uses the first-visit
 * Monte Carlo method. When `true`, only the first occurrence of each state-action pair
 * in a trajectory is used for updates.
 */
class OnPolicyMonteCarloControl(
    onPolicyUpdate: PolicyUpdate<Int, Int>,
    rng: Random = Random.Default,
    initialPolicy: Policy<Int, Int>,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate,
    private val gamma: Double,
    private val firstVisitOnly: Boolean,
) : TrajectoryLearningAlgorithm<Int, Int>(initialPolicy, onPolicyUpdate, rng) {
    private val N: MutableMap<Long, Int> = mutableMapOf()

    /**
     * Observes a trajectory of state-action-reward transitions and updates the Q-function
     * using the Monte Carlo control method. This method processes the trajectory in reverse order,
     * calculates the return (G), and updates Q-values for the state-action pairs. It allows
     * optional processing of a state-action pair only on its first visit for optimized updates.
     *
     * @param trajectory the sequence of state-action-reward transitions representing the experience
     *                   gathered in an episode.
     * @param episode the current episode number, often used for adaptive parameter schedules or
     *                episode-based logic.
     */
    override fun observe(trajectory: Trajectory<Int, Int>, episode: Int) {
        val seen = mutableSetOf<Long>()
        fun key(s: Int, a: Int) = (s.toLong() shl 32) xor (a.toLong() and 0xffffffffL)
        var G = 0.0

        for ((state, action, reward) in trajectory.asReversed()) {
            G = reward + gamma * G
            val sa = key(state, action)

            if (firstVisitOnly && !seen.add(sa)) continue

            N[sa] = (N[sa] ?: 0) + 1
            val oldQ = Q[state, action]
            Q[state, action] = oldQ + (G - oldQ) / N[sa]!!
        }
        onQUpdate(Q)
    }
}

