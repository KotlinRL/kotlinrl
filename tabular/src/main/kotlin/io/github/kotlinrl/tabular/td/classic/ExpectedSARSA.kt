package io.github.kotlinrl.tabular.td.classic

import io.github.kotlinrl.core.PolicyUpdate
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithm.*
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.random.*

/**
 * Implements the Expected SARSA (State-Action-Reward-State-Action) algorithm for reinforcement learning.
 *
 * Expected SARSA is an on-policy temporal difference learning algorithm, which extends SARSA by calculating
 * the expected value of the next state's Q-values using the policy's probability distribution over actions,
 * rather than sampling a single next action. This approach reduces variance and often improves stability.
 *
 * The algorithm updates the state-action value function (Q-table) iteratively using the following formula:
 * Q(s, a) = Q(s, a) + α * (r + γ * E[Q(s', a')] - Q(s, a)),
 * where:
 * - Q(s, a) represents the current state-action value,
 * - α is the learning rate,
 * - r is the reward obtained,
 * - γ is the discount factor,
 * - E[Q(s', a')] is the expected Q-value for the next state under the policy.
 *
 * The class also supports callbacks for updating both the policy and the Q-table after every transition update.
 *
 * @param initialPolicy the initial policy to be used for selecting actions based on states.
 * @param onPolicyUpdate a callback function invoked to update the policy during the learning process.
 * @param rng a random number generator used for action selection and other stochastic processes.
 * @param Q the Q-table used for tracking estimated action values for each state-action pair.
 * @param onQUpdate a callback function invoked to update the Q-table during the learning process.
 * @param alpha a parameter schedule representing the learning rate, which dynamically adjusts over time.
 * @param gamma the discount factor, controlling the weight given to future rewards.
 */
class ExpectedSARSA(
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random,
    epsilon: ParameterSchedule,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate = {},
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TransitionLearningAlgorithm<Int, Int>(Q.epsilonGreedy(epsilon, rng), onPolicyUpdate, rng) {
    /**
     * Observes a transition in the environment and updates the Q-value function based on the
     * expected SARSA (State-Action-Reward-State-Action) algorithm.
     *
     * This method calculates the expected Q-value by considering all possible next actions
     * in the given state, weighted by their probabilities as determined by the policy.
     * It updates the Q-value for the corresponding state-action pair using the update rule.
     *
     * If the transition is terminal, the expected Q-value for future rewards is set to zero.
     * After updating the Q-value table, a callback function is invoked to notify that
     * the table has been modified.
     *
     * @param transition The transition encapsulating a step in the reinforcement learning
     * environment, including the current state, action taken, reward received, the next state,
     * and whether the episode terminated.
     */
    override fun observe(transition: Transition<Int, Int>) {
        val (state, action, reward, sPrime, _, _, isTerminal) = transition
        val (alpha) = alpha()
        val distribution = policy[sPrime]
        val expectedQ = if (isTerminal) 0.0
        else Q[sPrime].mapIndexed { aPrime, _ ->
            distribution.prob(aPrime) * Q[sPrime, aPrime]
        }.max() ?: 0.0
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * expectedQ - Q[state, action])
        onQUpdate(Q)
    }
}