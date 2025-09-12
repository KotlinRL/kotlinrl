package io.github.kotlinrl.tabular.td.classic

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithm.TransitionLearningAlgorithm
import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.tabular.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*


/**
 * TDZero implements the TD(0) algorithm, a model-free reinforcement learning approach
 * that estimates the state-value function (V) based on one-step temporal differences.
 * It updates the value function incrementally after each observed transition in an episode.
 *
 * @param initialPolicy the initial policy used to determine actions in states.
 * @param onPolicyUpdate a callback function invoked whenever the policy is updated.
 * @param rng a random number generator for stochasticity in the learning process.
 * @param V the state-value function to be updated during learning.
 * @param onVUpdate a callback function invoked whenever the V-table is updated.
 * @param alpha a schedule representing the learning rate, which determines the step size
 *              in the update of the value function.
 * @param gamma the discount factor that defines how future rewards are weighted relative
 *              to immediate rewards.
 */
class TDZero(
    initialPolicy: Policy<Int, Int>,
    onPolicyUpdate: PolicyUpdate<Int, Int>,
    rng: Random,
    private val V: VTable,
    private val onVUpdate: VTableUpdate,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TransitionLearningAlgorithm<Int, Int>(initialPolicy, onPolicyUpdate, rng) {

    /**
     * Observes a state transition and updates the state-value function (V) using the TD(0) algorithm.
     *
     * Processes the given transition to compute a one-step temporal difference update for the value
     * associated with the current state. If the transition leads to a terminal state, the future reward
     * is set to zero. The updated value function is then passed to the associated callback.
     *
     * @param transition A transition object representing the current state, action, reward,
     * next state, and episode termination status. The transition includes details necessary
     * for calculating the temporal difference update.
     */
    override fun observe(transition: Transition<Int, Int>) {
        val (s, _, reward, sPrime, _, _, isTerminal) = transition
        val nextV = if (isTerminal) 0.0 else V[sPrime]
        val alpha = alpha().current
        V[s] = V[s] + alpha * (reward + gamma * nextV - V[s])
        onVUpdate(V)
    }
}