package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * Represents a functional interface for computing Temporal Difference (TD) errors for value functions.
 *
 * Temporal Difference methods are widely used in reinforcement learning to estimate value functions
 * by bootstrapping, which combines observed rewards with estimates of future values.
 * This interface provides a blueprint for defining error calculation logic in TD-based approaches,
 * such as TD(0), TD(λ), or other variations.
 *
 * @param State The type representing the states in the environment or Markov Decision Process (MDP).
 */
fun interface TDVError<State> {
    /**
     * Computes the Temporal Difference (TD) error for a given value function and transition.
     *
     * This method calculates the TD error, which represents the discrepancy between the estimated value
     * of the current state and the target value. The target value is based on the observed reward and
     * the estimated value of the subsequent state, scaled by the discount factor.
     *
     * @param V the value function used to retrieve the scalar value of a given state.
     * @param t the transition, containing the current state, reward, next state,
     * and a flag indicating if the episode has ended.
     * @param gamma the discount factor, a value between 0 and 1 that weighs the importance
     * of future rewards.
     * @return the computed TD error as a `Double`, capturing the difference between the predicted
     * and target state values.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        t: Transition<State, *>,
        gamma: Double
    ): Double
}
