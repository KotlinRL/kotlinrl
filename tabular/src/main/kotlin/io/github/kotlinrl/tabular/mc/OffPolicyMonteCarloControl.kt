package io.github.kotlinrl.tabular.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.agent.Trajectory
import io.github.kotlinrl.core.algorithm.TrajectoryLearningAlgorithm
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * Implements the Off-Policy Monte Carlo control algorithm for reinforcement learning.
 * This algorithm updates the Q-function and derives the optimal policy by processing
 * complete trajectories of state-action-reward transitions using importance sampling.
 * It supports first-visit updates and allows for the integration of a target policy.
 * It uses weighted importance sampling to estimate the return of a state-action pair.
 *
 * @property onPolicyUpdate A callback invoked whenever the policy is updated, enabling external
 *                          handling of policy modifications.
 * @property rng An instance of `Random` used for generating random numbers, particularly for
 *               epsilon-greedy action selection.
 * @property initialTargetPolicy The initial target policy to be improved during learning.
 * @property epsilon An exploration parameter managed by a dynamically adjustable schedule,
 *                   typically used in epsilon-greedy policies.
 * @property onPTableUpdate A callback invoked whenever the target policy table is updated,
 *                          allowing external handling of policy table changes.
 * @property Q The Q-table that maintains state-action value estimates.
 * @property onQUpdate A callback invoked whenever the Q-table is updated to support external
 *                     tracking of changes.
 * @property gamma The discount factor, determining the present value of future rewards.
 * @property firstVisitOnly A boolean indicating whether Q-value updates should be restricted
 *                          to the first visit of each state-action pair within an episode.
 */
class OffPolicyMonteCarloControl(
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random = Random.Default,
    initialTargetPolicy: PTable,
    private val epsilon: ParameterSchedule,
    private val onPTableUpdate: PTableUpdate = {},
    private val Q: QTable,
    private val onQUpdate: QTableUpdate = {},
    private val gamma: Double,
    private val firstVisitOnly: Boolean,
) : TrajectoryLearningAlgorithm<Int, Int>(Q.epsilonGreedy(epsilon, rng), onPolicyUpdate, rng) {
    private val C: MutableMap<Long, Double> = mutableMapOf()
    private val targetPolicy: PTable = initialTargetPolicy

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
        var W = 1.0

        data class StepWithProb(val s: Int, val a: Int, val r: Double, val bProb: Double)
        val steps = trajectory.map { (state, action, reward) ->
            StepWithProb(state, action, reward, policy[state].prob(action))
        }

        for ((state, action, reward, bProb) in steps.asReversed()) {
            G = reward + gamma * G
            val sa = key(state, action)

            if (firstVisitOnly && !seen.add(sa)) continue

            val newC = (C[sa] ?: 0.0) + W
            C[sa] = newC
            val oldQ = Q[state, action]
            Q[state, action] = oldQ + (W / newC) * (G - oldQ)
            val aGreedy = Q[state].argMax()
            if (targetPolicy[state] != aGreedy) {
                targetPolicy[state] = aGreedy
                onPTableUpdate(targetPolicy)
            }
            if (action != aGreedy) break
            if (bProb == 0.0) break
            W /= bProb
            if (!W.isFinite() || W > 1e12) break
        }
        onQUpdate(Q)
    }
}

