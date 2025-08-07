package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * An object providing utility methods for calculating Temporal Difference (TD) errors related to value functions.
 *
 * Temporal Difference learning is a key component of reinforcement learning algorithms, used to estimate
 * value functions. This object includes methods for both on-policy and off-policy TD(0) error calculations.
 */
object TDVErrors {
    /**
     * Computes the TD(0) error for a given value function and transition using an on-policy approach.
     *
     * The TD(0) error is the difference between the predicted state value and the observed return,
     * incorporating the immediate reward and the discounted estimated value of the subsequent state.
     * This method is a foundational component of Temporal Difference learning, enabling incremental
     * updates to the value function based on sequential transitions in the state space.
     *
     * @return a `TDVError<State>` functional object that, when invoked, computes the temporal difference
     *         error for the provided value function, transition, and discount factor.
     */
    // On-policy TD(0) for V
    fun <State> tdZero(): TDVError<State> =
        TDVError { V, t, gamma ->
            val (s, _, r, sPrime) = t
            val boot = if (t.done) 0.0 else V[sPrime]
            r + gamma * boot - V[s]
        }

    /**
     * Constructs a TD(0) error computation function for value prediction utilizing per-step importance weights.
     * This TD(0) is an off-policy variant, which adjusts the Temporal Difference (TD) error using an
     * importance weight `rho` specific to each transition.
     *
     * The importance weight `rho` allows the adjustment of the target value based on the difference
     * between the behavior policy (that generates the data) and the target policy (for which value is estimated).
     *
     * @param rho a function mapping a transition to its importance weight. Defaults to `1.0`, implying no weighting.
     * @return a `TDVError` instance configured to compute the weighted TD(0) error for value prediction.
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