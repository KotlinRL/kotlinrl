package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * A functional interface used to compute Temporal Difference (TD) errors for Q-functions.
 * The TD error is an indication of the discrepancy between the predicted Q-value of a state-action pair
 * and the target Q-value, which incorporates the observed reward and the Q-value of subsequent states.
 *
 * This interface provides the structure for implementing different TD error calculation strategies
 * such as Q-Learning, SARSA, Expected SARSA, and semi-gradient methods.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions in the environment.
 */
fun interface TDQError<State, Action> {
    /**
     * Computes the Temporal Difference (TD) error for a given Q-function and transition.
     *
     * This function calculates the TD error, which represents the difference between
     * the predicted Q-value of the current state-action pair and the target Q-value
     * that incorporates observed reward and the next state-action value. The computation
     * depends on the specific TD method being used (e.g., Q-Learning, SARSA, Expected SARSA).
     *
     * @param Q the Q-function to be used for retrieving Q-values of state-action pairs.
     * @param t the transition, containing the current state, action, reward, next state,
     * and a flag indicating if the episode has ended.
     * @param aPrime the action taken in the subsequent state (used in on-policy methods
     * like SARSA). Can be null for off-policy methods like Q-Learning.
     * @param gamma the discount factor, a value between 0 and 1, that weighs the importance
     * of future rewards.
     * @param done a flag indicating if the episode has ended after the current transition.
     * This determines whether future states are considered.
     * @return the computed TD error as a `Double`, capturing the deviation between the
     * predicted and target Q-values.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        t: Transition<State, Action>,
        aPrime: Action?,
        gamma: Double,
        done: Boolean
    ): Double
}
