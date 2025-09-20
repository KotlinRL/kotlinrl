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
 * Implements the Q-Learning algorithm for reinforcement learning tasks.
 *
 * Q-Learning is an off-policy temporal-difference control algorithm used to learn the
 * optimal action-value function, enabling an agent to act optimally in a given environment.
 * The algorithm updates Q-values based on transitions observed during interaction with the
 * environment, using a Bellman equation with a learning rate (alpha) and a discount factor (gamma).
 *
 * This implementation supports:
 * - Epsilon-greedy exploration, governed by a schedule for dynamically adjusting the value of epsilon.
 * - Customizable hooks for handling policy updates and Q-value updates when changes occur.
 *
 * @param onPolicyUpdate Function invoked when the policy is updated. Can be used to monitor or log
 * policy updates. Defaults to an empty function.
 * @param rng Random number generator used for exploration actions during epsilon-greedy policy.
 * @param epsilon A schedule governing the value of epsilon for the epsilon-greedy exploration strategy.
 * @param Q Q-value table representing the state-action value function.
 * @param onQUpdate Function invoked when Q-values are updated. Can be used for monitoring, logging, or other side effects.
 * @param alpha A schedule for dynamically adjusting the learning rate (alpha) during training.
 * @param gamma The discount factor used to weigh future rewards. Defines the agent's consideration for long-term rewards relative to immediate rewards.
 */
class QLearning(
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random = Random.Default,
    epsilon: ParameterSchedule,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate = {},
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TransitionLearningAlgorithm<Int, Int>(Q.epsilonGreedy(epsilon, rng), onPolicyUpdate, rng) {

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
        val (state, action, reward, sPrime) = transition
        val (alpha) = alpha()
        Q[state, action] = Q[state, action] + alpha * (reward + (gamma * (Q[sPrime].max() ?: 0.0)) - Q[state, action])
        onQUpdate(Q)
    }
}