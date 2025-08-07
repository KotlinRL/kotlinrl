package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Represents a functional interface for calculating the n-step Temporal Difference (TD) value error.
 *
 * This interface is designed to compute the discrepancy between the estimated value of a state
 * and the observed return over n steps in a reinforcement learning context. It utilizes a user-defined
 * value function, an observed trajectory, and a discount factor to determine the error.
 *
 * The n-step TD value error is a critical component in reinforcement learning algorithms,
 * enabling updates to value functions based on multi-step lookahead with discounted rewards.
 *
 * @param State The type representing the states within the environment.
 */
fun interface NStepTDVError<State> {
    /**
     * Calculates the n-step Temporal Difference (TD) value error.
     *
     * This method computes the discrepancy between the observed n-step return in a trajectory
     * and the estimated value of a state. It incorporates a value function, a trajectory, and
     * a discount factor to calculate the TD error in reinforcement learning scenarios.
     *
     * @param V The value function used to estimate the value of states.
     * @param t The trajectory containing state-action-reward transitions.
     * @param gamma The discount factor applied to future rewards, in the range [0, 1].
     * @return The computed n-step TD value error.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        t: Trajectory<State, *>,
        gamma: Double
    ): Double
}
