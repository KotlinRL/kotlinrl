package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements an n-step Temporal Difference (TD) learning method for updating a value function
 * based on a given trajectory of states, actions, and rewards.
 *
 * This class performs updates to an enumerable scalar value function using n-step TD updates,
 * where the error is computed over multiple steps in the trajectory. The update is controlled
 * by a step-size parameter (`alpha`) and a discount factor (`gamma`). It provides a balance
 * between bias and variance by utilizing information over multiple steps, making it suitable
 * for reinforcement learning tasks.
 *
 * The class relies on the `NStepTDVError` interface to compute the n-step TD error, enabling
 * extensibility in the way errors are computed. An optional `NStepTDVError` implementation can
 * be provided; otherwise, a default implementation is used.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param alpha a parameter schedule controlling the step size for updating the value function.
 * @param gamma the discount factor applied to future rewards, in the range [0, 1].
 * @param td an instance of `NStepTDVError` for computing the n-step TD value error. Defaults to a standard implementation.
 */
class NStepEstimateV_fromTrajectory<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: NStepTDVError<State> = NStepTDVErrors.nStep()
) : EstimateV_fromTrajectory<State, Action> {

    /**
     * Updates the value function using the n-step Temporal Difference (TD) learning algorithm.
     *
     * The method applies an update to the input value function based on the given trajectory of
     * states, actions, and rewards. If the trajectory is empty, the method returns the original
     * value function. Otherwise, it calculates the TD error and updates the value of the initial
     * state in the trajectory based on the computed error, step-size parameter (`alpha`), and the
     * current state value. The updated value function is then returned.
     *
     * @param V the value function to be updated, representing the mapping from states to their values.
     * @param trajectory the trajectory of states, actions, and rewards used to compute the TD error
     *                   and update the value function.
     * @return the updated value function after applying the n-step TD update. If no update is applied
     *         (e.g., empty trajectory or zero TD error), the original value function is returned.
     */
    override fun invoke(V: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        if (trajectory.isEmpty()) return V

        val s0 = trajectory.first().state
        val delta = td(V, trajectory, gamma)
        if (delta == 0.0) return V
        return V.update(s0, V[s0] + alpha() * delta)
    }
}
