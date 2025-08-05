package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*
import kotlin.math.pow

/**
 * Provides methods for calculating n-step Temporal Difference (TD) value errors in
 * reinforcement learning scenarios.
 *
 * These methods compute the error between predicted state values and actual returns
 * derived from observed trajectories. They are essential in n-step temporal difference
 * learning for policy evaluation or improvement strategies, helping to improve the accuracy
 * of the value function by incorporating multiple steps of a trajectory.
 */
object NStepTDVErrors {
    /**
     * Creates an [NStepTDVError] function to compute the n-step Temporal Difference (TD) value error
     * for a given value function, trajectory, and discount factor.
     *
     * The returned function calculates the difference between the observed rewards (adjusted by the
     * discount factor) over a trajectory and the predicted value of the initial state. If the
     * trajectory ends in a non-terminal state, the discounted estimated value for the next state is used.
     *
     * The n-step TD error provides a balance between bias and variance by incorporating information
     * from multiple steps, which is a critical component for reinforcement learning algorithms.
     *
     * @return an instance of [NStepTDVError] that computes the n-step TD value error for any state type.
     */
    fun <State> nStep(): NStepTDVError<State> =
        NStepTDVError { V, traj, gamma ->
            if (traj.isEmpty()) return@NStepTDVError 0.0
            val s0 = traj.first().state
            val terminal = traj.last().done

            var G = 0.0
            for ((i, t) in traj.withIndex()) {
                G += gamma.pow(i) * t.reward
            }
            if (!terminal) {
                val sT = traj.last().nextState
                G += gamma.pow(traj.size) * V[sT]
            }
            G - V[s0]
        }

    /**
     * Creates a weighted/off-policy variant of the n-step Temporal Difference (TD) value error computation.
     *
     * This function takes a weighting function `rho` and combines it with the standard n-step TD error computation
     * to enable off-policy learning scenarios where the observed trajectories are weighted differently. The returned
     * function computes the weighted n-step TD value error for a given value function, trajectory, and discount factor.
     *
     * @param rho a weighting function that takes a trajectory and returns a weight (importance sampling ratio)
     *            to be applied to the n-step TD error computation.
     * @return an instance of [NStepTDVError] that computes the weighted n-step TD value error.
     */
    // Optional: weighted/off-policy variant
    fun <State> nStepWeighted(
        rho: (Trajectory<State, *>) -> Double
    ): NStepTDVError<State> =
        NStepTDVError { V, traj, gamma ->
            rho(traj) * nStep<State>()(V, traj, gamma)
        }
}