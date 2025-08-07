package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * TDQErrors provides a collection of standard Temporal Difference (TD) error calculation functions
 * for reinforcement learning, enabling the computation of TD errors for Q-functions using methods
 * such as Q-Learning, SARSA, and Expected SARSA.
 *
 * These functions facilitate the comparison of predicted Q-values against target values that
 * incorporate observed rewards and future estimated Q-values, serving as a cornerstone
 * for updating Q-functions in various RL algorithms.
 */
object TDQErrors {
    /**
     * Creates a Temporal Difference (TD) error function for the Q-Learning algorithm.
     *
     * Q-Learning is an off-policy reinforcement learning algorithm used to estimate the
     * optimal action-value function (Q-function). This function defines the calculation
     * of the TD error, which measures the discrepancy between the predicted Q-value for
     * a state-action pair and a target Q-value computed using the Bellman optimality equation.
     *
     * The Q-Learning TD error is formulated as:
     * TD error = reward + γ * max_a' Q(next_state, a') - Q(state, action)
     * where `max_a' Q(next_state, a')` is the maximum Q-value for all possible actions
     * in the next state, and γ is the discount factor.
     *
     * @return a `TDQError` instance representing the Q-Learning TD error function.
     */
    fun <State, Action> qLearning(): TDQError<State, Action> =
        TDQError { Q, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q.maxValue(sPrime)
            r + gamma * nextQ - Q[s, a]
        }

    /**
     * Constructs the TD error computation for the SARSA (State-Action-Reward-State-Action) algorithm.
     *
     * The SARSA TD error is calculated using the observed reward, the current Q-value of the state-action pair,
     * and the Q-value of the next state-action pair. It is an on-policy TD learning method that incorporates
     * the action taken in the next state when computing the target value. If the transition represents the
     * end of an episode, the TD target incorporates no future Q-value.
     *
     * @return a `TDQError` instance configured to compute the TD error following the SARSA algorithm.
     * The computed TD error quantifies the discrepancy between the Q-value prediction and the on-policy target.
     */
    fun <State, Action> sarsa(): TDQError<State, Action> =
        TDQError { Q, t, aPrime, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q[sPrime, aPrime!!]
            r + gamma * nextQ - Q[s, a]
        }

    /**
     * Computes the Temporal Difference (TD) error using the Expected SARSA method.
     *
     * This method combines the predicted Q-value of the current state-action pair, reward,
     * and the expected Q-value of the subsequent state to calculate the error. The expected
     * Q-value is computed using the policy's probabilities of selecting actions in the next state.
     *
     * @param State the type representing the states in the environment.
     * @param Action the type representing the actions in the environment.
     * @param policy the policy used to determine the probabilities of actions for the next state.
     *               The policy also provides a Q-function that maps state-action pairs to Q-values.
     * @return a TDQError instance that computes the Expected SARSA TD error using the given policy.
     */
    fun <State, Action> expectedSarsa(policy: Policy<State, Action>): TDQError<State, Action> =
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