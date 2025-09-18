package io.github.kotlinrl.tabular.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.agent.Trajectory
import io.github.kotlinrl.core.algorithm.TrajectoryLearningAlgorithm
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * Implements the Incremental Monte Carlo Control algorithm for reinforcement learning.
 * This algorithm estimates the optimal action-value function (Q-function) based on trajectories
 * of state-action-reward transitions, using a Monte Carlo approach with optional first-visit control.
 * The Q-function is updated incrementally for better computational efficiency.
 *
 * @constructor Creates an instance of the IncrementalMonteCarloControl algorithm.
 *
 * @param onPolicyUpdate A callback function invoked whenever the policy is updated during learning.
 * @param rng A random number generator for stochastic processes, such as action selection.
 * @param epsilon A schedule for the epsilon parameter used in epsilon-greedy exploration.
 * @param Q The action-value function (Q-table) used to store and update state-action values.
 * @param onQUpdate A callback function invoked whenever the Q-function is updated.
 * @param alpha A schedule for the learning rate parameter, which controls the step size for updates.
 * @param gamma The discount factor for future rewards, representing the weight of future rewards relative to immediate rewards.
 * @param firstVisitOnly If true, Q-function updates are performed only on the first visit to state-action pairs
 *                       within a trajectory (First-visit Monte Carlo). If false, every visit is considered (Every-visit Monte Carlo).
 */
class IncrementalMonteCarloControl(
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random = Random.Default,
    epsilon: ParameterSchedule,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate = {},
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val firstVisitOnly: Boolean,
) : TrajectoryLearningAlgorithm<Int, Int>(Q.epsilonGreedy(epsilon, rng), onPolicyUpdate, rng) {

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

            val (alpha) = alpha()
            val oldQ = Q[state, action]
            Q[state, action] = oldQ + alpha * (G - oldQ)
        }
        onQUpdate(Q)
    }
}
