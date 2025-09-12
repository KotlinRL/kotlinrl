package io.github.kotlinrl.tabular.mc

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithm.TrajectoryLearningAlgorithm
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.api.ParameterSchedule.Companion.constant
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * The `IncrementalMonteCarloControl` class implements a reinforcement learning algorithm
 * based on Monte Carlo control with incremental updates to the Q-function. This algorithm
 * processes state-action-reward trajectories and adjusts Q-values accordingly to improve
 * the policy over time.
 *
 * The key features of this class include:
 * - Incremental updates to the Q-function using the provided learning rate (`alpha`) and
 *   discount factor (`gamma`).
 * - Optional focus on first-visit state-action pairs only.
 * - Support for custom policy update mechanisms and Q-value update callbacks.
 *
 * The algorithm observes complete trajectories from interactions with the environment,
 * calculates long-term returns (rewards) for each state-action pair, and updates the
 * Q-function incrementally, which is then utilized to improve the policy.
 *
 * @param initialPolicy the initial policy used for action selection. This is typically
 * determined at the start of training and updated over time based on the Q-function.
 * @param onPolicyUpdate a callback method for updating the policy when significant changes
 * in the Q-function are detected.
 * @param rng a random number generator used for stochastic operations in the algorithm,
 * with a default value of `Random.Default`.
 * @param Q a reference to the Q-function (or Q-table) being updated throughout the learning
 * process. This represents the estimated value of state-action pairs.
 * @param onQUpdate a callback invoked whenever the Q-function is updated. Useful for
 * monitoring or additional processing after Q-table adjustments.
 * @param alpha the schedule for the learning rate. This influences how much new information
 * affects previously learned Q-values. Can be static or dynamically adjusted over time.
 * @param gamma the discount factor applied to future rewards. Determines the importance
 * of long-term versus short-term rewards in the value estimation.
 * @param firstVisitOnly a flag indicating whether only the first occurrence of each unique
 * state-action pair in a trajectory should be considered for Q-value updates. If `true`,
 * the algorithm updates only first-visit occurrences; otherwise, all occurrences are processed.
 */
class IncrementalMonteCarloControl(
    initialPolicy: Policy<Int, Int>,
    onPolicyUpdate: PolicyUpdate<Int, Int>,
    rng: Random = Random.Default,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val firstVisitOnly: Boolean,
) : TrajectoryLearningAlgorithm<Int, Int>(initialPolicy, onPolicyUpdate, rng) {

    /**
     * Observes a trajectory and updates the Q-function based on the Monte Carlo control strategy.
     *
     * @param trajectory the trajectory of state-action-reward transitions to be processed.
     * @param episode the current episode number, useful for adaptive parameter schedules or episode-based logic.
     */
    override fun observe(trajectory: Trajectory<Int, Int>, episode: Int) {
        val seen = mutableSetOf<Long>()
        fun key(s: Int, a: Int) = (s.toLong() shl 32) xor (a.toLong() and 0xffffffffL)
        var G = 0.0

        for ((state, action, reward) in trajectory.asReversed()) {
            G = reward + gamma * G

            if (firstVisitOnly && !seen.add(key(state, action))) continue

            val alpha = alpha().current
            val oldQ = Q[state, action]
            Q[state, action] = oldQ + alpha * (G - oldQ)
        }
        onQUpdate(Q)
    }
}
