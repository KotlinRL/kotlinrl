package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Represents a function for computing the n-step Temporal Difference (TD) value error
 * for a given value function in reinforcement learning.
 *
 * The n-step TD value error is a core component in reinforcement learning algorithms
 * like n-step TD methods. It calculates the error between the value function's predictions
 * and the observed rewards over a trajectory, with a discount applied. This error serves
 * as the foundation for updating the value function to improve its predictions over time.
 *
 * n-Step approaches balance bias and variance by incorporating information from multiple
 * steps of a trajectory, rather than just the immediate step, enabling more stable learning
 * in some environments.
 *
 * @param State The type representing the environment's state space.
 */
fun interface NStepTDVError<State> {
    /**
     * Computes the n-step Temporal Difference (TD) value error for a given value function
     * using the observed trajectory and a specified discount factor.
     *
     * @param V the value function used to estimate state values.
     * @param t the trajectory containing the sequence of states, actions, and rewards observed.
     * @param gamma the discount factor applied to future rewards, in the range [0, 1].
     * @return the computed n-step TD value error.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        t: Trajectory<State, *>,
        gamma: Double
    ): Double
}
