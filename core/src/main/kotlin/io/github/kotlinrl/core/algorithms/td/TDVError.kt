package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

/**
 * A functional interface for computing Temporal Difference (TD) errors based on state values.
 *
 * The TD error quantifies the difference between the estimated value of the current state
 * and the target value derived from the observed reward and the estimated value of the subsequent state.
 * This serves as a foundational component in reinforcement learning algorithms for value prediction,
 * enabling incremental updates towards more accurate value functions.
 *
 * This interface provides flexibility in defining various TD error calculation strategies,
 * such as standard TD error computation or methods tailored to specific environments or learning objectives.
 *
 * @param State The type representing the states in the environment.
 */
fun interface TDVError<State> {
    /**
     * Computes the Temporal Difference (TD) error based on the provided value function and transition.
     *
     * This operator function calculates the TD error as the discrepancy between the observed target value,
     * which incorporates a discounted estimate of future rewards, and the estimated value of the current state.
     * The gamma parameter determines the discount factor applied to future rewards.
     *
     * @param V the value function that maps a state to its estimated value.
     * @param t the transition, which contains the current state, reward, next state, and a flag indicating
     *          whether the transition ends an episode.
     * @param gamma the discount factor, a value in the range [0, 1], which determines the trade-off
     *              between immediate and future rewards.
     * @return the calculated TD error as a `Double`, representing the difference between
     *         the predicted value and the target value.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        t: Transition<State, *>,
        gamma: Double
    ): Double
}
