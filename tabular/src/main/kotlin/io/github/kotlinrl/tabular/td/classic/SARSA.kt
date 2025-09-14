package io.github.kotlinrl.tabular.td.classic

import io.github.kotlinrl.core.PolicyUpdate
import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithm.TransitionLearningAlgorithm
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

/**
 * Implements the SARSA (State-Action-Reward-State-Action) algorithm for reinforcement learning,
 * which is an on-policy temporal difference control method. SARSA updates the Q-function based
 * on the experienced transitions and the action chosen by the current policy in the next state.
 *
 * @constructor Initializes a SARSA learning algorithm.
 * @param initialPolicy the initial policy used to decide actions based on states.
 * @param onPolicyUpdate the callback function to be invoked when the policy is updated.
 * @param rng the random number generator used by the algorithm for stochastic decisions.
 * @param Q the Q-value table that maps state-action pairs to their estimated values.
 * @param onQUpdate the callback function to be invoked when the Q-value table is updated.
 * @param alpha the learning rate parameter schedule, which controls how quickly the Q-values are updated.
 * @param gamma the discount factor, which determines the importance of future rewards.
 */
class SARSA(
    initialPolicy: Policy<Int, Int>,
    onPolicyUpdate: PolicyUpdate<Int, Int> = {},
    rng: Random,
    private val Q: QTable,
    private val onQUpdate: QTableUpdate = {},
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TransitionLearningAlgorithm<Int, Int>(initialPolicy, onPolicyUpdate, rng) {
    private var previous: Transition<Int, Int>? = null

    /**
     * Observes a transition in the environment and updates the Q-value function
     * based on the SARSA (State-Action-Reward-State-Action) algorithm.
     *
     * This method processes the given transition, combines it with the previous
     * state-action-reward transition (if available), and performs an on-policy update
     * to the Q-value for the corresponding state-action pair. If the transition is terminal,
     * the Q-value for future rewards is considered zero, and the previous transition is cleared.
     *
     * @param transition The transition representing the interaction between the
     * agent and the environment, including the current state, action taken,
     * reward received, next state, and whether the episode ended.
     */
    override fun observe(transition: Transition<Int, Int>) {
        val bootstrap = previous
        previous = transition

        if (bootstrap == null) return

        val (s, a, reward) = bootstrap
        val (sPrime, aPrime, _, _, _, _, isTerminal) = transition
        val nextQ = if (isTerminal) 0.0 else Q[sPrime, aPrime]
        val (alpha) = alpha()
        Q[s, a] = Q[s, a] + alpha * (reward + gamma * nextQ - Q[s, a])
        if (isTerminal) previous = null
        onQUpdate(Q)
    }
}