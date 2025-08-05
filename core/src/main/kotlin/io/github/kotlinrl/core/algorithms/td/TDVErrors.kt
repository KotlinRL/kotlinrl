package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * Provides implementations of Temporal Difference (TD) error calculations for value functions.
 *
 * `TDVErrors` contains methods for computing TD errors, which measure the discrepancy between the current
 * estimate of a state value and the target value. These methods are used in Reinforcement Learning
 * algorithms to iteratively improve value function estimations.
 */
object TDVErrors {
    /**
     * Implements on-policy Temporal Difference (TD(0)) for value function updates.
     *
     * This method returns a TDVError functional interface instance, which computes the TD error for
     * a value function based on the given state transition, observed rewards, and a discount factor.
     * In the TD(0) method, the value function is updated using the observed reward and the value
     * of the subsequent state, discounted by the provided factor.
     *
     * The TD error is computed as:
     * - If the transition is terminal: TD error = reward - value of current state.
     * - Otherwise: TD error = reward + (discount factor * value of subsequent state) - value of current state.
     *
     * @return a TDVError instance for calculating TD errors in the TD(0) method.
     */
    // On-policy TD(0) for V
    fun <State> tdZero(): TDVError<State> =
        TDVError { V, t, gamma ->
            val (s, _, r, sPrime) = t
            val boot = if (t.done) 0.0 else V[sPrime]
            r + gamma * boot - V[s]
        }

    /**
     * Constructs an off-policy Temporal Difference (TD(0)) error computation logic for value functions,
     * using per-step importance weighting.
     *
     * This method implements TD(0) for value function approximation, incorporating an optional importance
     * weight `rho` for each transition. The computed error represents the difference between the expected
     * and actual value of the current state, adjusted by observed rewards and the value of the next state.
     *
     * @param rho a function that computes the importance weight for a given transition. By default,
     * it returns 1.0 for all transitions.
     * @return an instance of `TDVError` that calculates the TD error for value functions using the provided
     * importance weighting and the TD(0) update rule.
     */
    // Off-policy TD(0) for V with per-step importance weight rho
    fun <State> tdZeroWeighted(
        rho: (Transition<State, *>) -> Double = { 1.0 }
    ): TDVError<State> =
        TDVError { V, t, gamma ->
            val (s, _, r, sPrime) = t
            val boot = if (t.done) 0.0 else V[sPrime]
            rho(t) * (r + gamma * boot - V[s])
        }
}