package io.github.kotlinrl.tabular.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * Implements an on-policy Monte Carlo control algorithm for reinforcement learning.
 *
 * The algorithm estimates optimal policies and action-value functions (Q-values)
 * by processing complete episodic experiences (trajectories). It updates Q-values
 * based on the rewards obtained over the entire trajectory, enabling improvements
 * in the policy that governs action selection.
 *
 * This implementation uses an epsilon-greedy policy for exploration during learning,
 * allowing a trade-off between exploration and exploitation. The exploration rate
 * (epsilon) is dynamically adjusted using a parameter schedule. The algorithm also
 * supports updates based on either every visit to a state-action pair or only the
 * first visit, depending on the configuration.
 *
 * @constructor Creates an instance of OnPolicyMonteCarloControl.
 *
 * @param onPolicyUpdate A callback invoked whenever the policy is updated. By default, it does nothing.
 *                       The callback can be customized to handle events triggered by policy updates.
 * @param rng The random number generator used for stochastic processes within the epsilon-greedy policy.
 *            Defaults to Kotlin's Random.Default.
 * @param epsilon A parameter schedule defining how the exploration rate evolves over time during learning.
 *                This schedule is used by the epsilon-greedy policy to balance exploration and exploitation.
 * @param Q The Q-table maintaining the action-value estimates. This table is updated using the Monte Carlo
 *          control method based on episodic rewards.
 * @param onQTableUpdate A callback invoked whenever the Q-table is updated. By default, it does nothing.
 *                       This callback can be used to monitor or log changes to the Q-values.
 * @param gamma The discount factor for future rewards. It determines the weight of future rewards when
 *              computing the return for a state-action pair.
 * @param firstVisitOnly Indicates whether the updates to the Q-function should be based only on the first
 *                       visit to each state-action pair in a trajectory. If `true`, only the first encounter
 *                       of each state-action pair is used for updates; otherwise, all visits are used.
 */
class OnPolicyMonteCarloControl(
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random = Random.Default,
    epsilon: ParameterSchedule,
    private val Q: QTable,
    private val onQTableUpdate: QTableUpdate = {},
    private val gamma: Double,
    private val firstVisitOnly: Boolean,
) : TrajectoryLearningAlgorithm<Int, Int>(initialPolicy = Q.epsilonGreedy(epsilon, rng), onPolicyUpdate, rng) {
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
        onQTableUpdate(Q)
    }
}

