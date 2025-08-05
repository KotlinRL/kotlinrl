package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements an estimator for updating a value function using n-step Temporal Difference (TD) learning.
 *
 * This class calculates updates to a provided enumerable value function based on a trajectory of
 * state-action transitions. The n-step TD method computes updates using a partial trajectory
 * and bootstraps from future estimates of the value function. The estimator applies a learning
 * rate (`alpha`), a discount factor (`gamma`), and allows customization of the temporal-difference
 * error computation through an `NStepTDVError` implementation.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param alpha a parameter schedule defining the learning rate used for value function updates.
 *              This dynamically adjusts the step size used during each update.
 * @param gamma the discount factor controlling the weight of future rewards in value function updates.
 *              A value between 0 and 1 determines how much future rewards influence the current value.
 * @param td an implementation of `NStepTDVError` to calculate the n-step temporal-difference error.
 *           The default implementation is a standard n-step TD error calculation.
 */
class NStepTDValueFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: NStepTDVError<State> = NStepTDVErrors.nStep()
) : TrajectoryValueFunctionEstimator<State, Action> {

    /**
     * Estimates the updated value function using n-step Temporal Difference (TD) learning
     * based on the given trajectory and the current value function.
     *
     * This method computes the temporal-difference error for the provided trajectory
     * and updates the value function for the initial state in the trajectory if
     * the error is non-zero. The update includes scaling by the learning rate (`alpha`)
     * and adjusting by the computed error.
     *
     * If the trajectory is empty or the temporal-difference error is zero, the method
     * returns the original value function without modification.
     *
     * @param V the current enumerable value function that maps states to their estimated values.
     * @param trajectory the trajectory consisting of a sequence of state-action-reward transitions,
     *                   used for computing the temporal-difference error and updating the value function.
     * @return the updated enumerable value function after applying the n-step TD learning update,
     *         or the original value function if no updates are applied.
     */
    override fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State> {
        if (trajectory.isEmpty()) return V

        val s0 = trajectory.first().state
        val delta = td(V, trajectory, gamma)
        if (delta == 0.0) return V
        return V.update(s0, V[s0] + alpha() * delta)
    }
}
