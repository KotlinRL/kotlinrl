package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * A collection of Temporal Difference (TD) error calculation strategies for Q-functions.
 *
 * This object provides implementations of commonly used TD error methods, tailored for different RL algorithms.
 * The calculated TD errors are used to update Q-values in reinforcement learning algorithms.
 * Methods include implementations for Q-Learning, SARSA, Expected SARSA, and semi-gradient SARSA.
 */
object TDQErrors {
    /**
     * Creates a Temporal Difference (TD) error function based on the Q-Learning algorithm.
     *
     * The resulting TD error function computes the error for a state-action pair,
     * where the target value is derived from the observed reward and the maximum Q-value
     * of the subsequent state. If the transition signifies the end of an episode, the target
     * value does not include future state values.
     *
     * This implementation is off-policy, as it uses the maximum Q-value across all possible actions
     * in the next state, irrespective of the action actually taken.
     *
     * @return a TDQError implementation that calculates TD errors using the Q-Learning method.
     */
    fun <State, Action> qLearning(): TDQError<State, Action> =
        TDQError { Q, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q.maxValue(sPrime)
            r + gamma * nextQ - Q[s, a]
        }

    /**
     * Constructs a Temporal Difference (TD) error calculation strategy using the SARSA algorithm.
     *
     * SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.
     * It computes the TD error by considering the reward received after transitioning
     * to the next state and the Q-value of the next state-action pair under the current policy.
     *
     * The computed TD error is used to update the Q-function and improve the policy.
     * This implementation accounts for terminal states by setting the future Q-value
     * to zero if the episode has ended.
     *
     * @return a `TDQError` functional interface that calculates the SARSA TD error
     * for a given Q-function and transition, using the provided discount factor and
     * next action under the current policy.
     */
    fun <State, Action> sarsa(): TDQError<State, Action> =
        TDQError { Q, t, aPrime, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q[sPrime, aPrime!!]
            r + gamma * nextQ - Q[s, a]
        }

    /**
     * Computes the Temporal Difference (TD) error for Expected SARSA using the supplied policy.
     *
     * This function calculates the TD error based on the Expected SARSA algorithm, which utilizes
     * a probabilistic expectation over next-state Q-values to determine the target Q-value.
     * The TD error is then used to update the Q-function for better action-value estimation.
     *
     * @param policy the Q-function-based policy that provides the probabilities of selecting
     * each action in a given state and the Q-values to calculate the TD error.
     * @return a TDQError functional interface instance that computes the TD error given a Q-function,
     * a state-action transition, discount factor, and episode completion flag.
     */
    fun <State, Action> expectedSarsa(policy: QFunctionPolicy<State, Action>): TDQError<State, Action> =
        TDQError { _, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val Q = policy.Q
            val expectedQ = if (!t.done) {
                policy.probabilities(sPrime).entries.sumOf { (aPrime, prob) ->
                    prob * Q[sPrime, aPrime]
                }
            } else 0.0
            r + gamma * expectedQ - Q[s, a]
        }
}