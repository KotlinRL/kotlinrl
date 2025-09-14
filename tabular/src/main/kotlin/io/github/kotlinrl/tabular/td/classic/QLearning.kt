package io.github.kotlinrl.tabular.td.classic

import io.github.kotlinrl.core.PolicyUpdate
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithm.*
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.random.*

/**
 * Implementation of the Q-Learning algorithm for reinforcement learning.
 *
 * Q-Learning is an off-policy temporal-difference learning algorithm that learns the value
 * of actions in a given state. The algorithm updates the Q-value for a state-action pair based
 * on the observed reward and the maximum Q-value of the subsequent state, using the following equation:
 *
 * Q(s, a) <- Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))
 *
 * where:
 * - Q(s, a) is the current Q-value for state `s` and action `a`.
 * - α (alpha) is the learning rate that controls the extent to which newly acquired
 *   information overrides the previous Q-value.
 * - γ (gamma) is the discount factor, determining the importance of future rewards.
 * - r is the observed reward after taking action `a` in state `s`.
 * - max_a' Q(s', a') is the maximum Q-value for actions in the next state `s'`.
 *
 * @constructor Creates an instance of the QLearning class.
 * @param initialPolicy the initial decision-making policy used to select actions in each state.
 * @param onPolicyUpdate callback function called when the policy is updated.
 * @param rng the random number generator used in the algorithm.
 * @param Q the Q-value table representing the value estimates for each state-action pair.
 * @param onQUpdate callback function called when the Q-value table is updated.
 * @param alpha the schedule controlling the learning rate (α) during the algorithm's execution.
 * @param gamma the discount factor (γ), determining the importance of future rewards.
 */
class QLearning(
    initialPolicy: Policy<Int, Int>,
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate = {},
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TransitionLearningAlgorithm<Int, Int>(initialPolicy, onPolicyUpdate, rng) {

    /**
     * Observes a transition in the environment and updates the Q-value function
     * based on the Q-Learning algorithm.
     *
     * This method processes the given transition, calculates the temporal-difference
     * error, and updates the Q-value for the corresponding state-action pair.
     * If the transition is terminal, future rewards are assumed to be zero.
     *
     * @param transition The transition representing the interaction between the
     * agent and the environment, including the current state, action taken, reward received,
     * next state, and whether the episode ended.
     */
    override fun observe(transition: Transition<Int, Int>) {
        val (state, action, reward, sPrime, _, _, isTerminal) = transition
        val (alpha) = alpha()
        val maxQ = if (isTerminal) 0.0 else Q[sPrime].max() ?: 0.0
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * maxQ - Q[state, action])
        onQUpdate(Q)
    }
}