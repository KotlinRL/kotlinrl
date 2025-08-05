package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements an n-step Temporal Difference (TD) Q-function estimator for reinforcement learning.
 *
 * This class estimates an updated Q-function by applying the n-step TD error to a trajectory of
 * state-action transitions. It is designed for use in reinforcement learning algorithms that
 * perform value function updates over multiple steps, such as n-step SARSA or n-step Q-learning.
 *
 * The estimator processes a given trajectory, computes the TD error based on the n-step temporal
 * difference update rule, and applies the computed error to update the Q-values for the
 * corresponding state-action pair.
 *
 * @param State the type representing the environment's state space.
 * @param Action the type representing the available actions in the environment.
 * @param initialPolicy an optional policy used to guide behavior and calculate expected Q-values
 *        for stochastic environments. Default is null.
 * @param alpha a parameter schedule defining the learning rate for updating Q-values.
 * @param gamma the discount factor applied to future rewards, determining the trade-off between
 *        immediate and future rewards. Value must be in the range [0, 1].
 * @param td a function for computing the n-step TD error based on the Q-function, trajectory,
 *        policy, tail action, and gamma.
 */
class NStepTDQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>? = null,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: NStepTDQError<State, Action>
) : TrajectoryQFunctionEstimator<State, Action> {

    var policy = initialPolicy
    var tailAction: Action? = null

    /**
     * Estimates and updates the given Q-function using n-step Temporal Difference (TD) learning.
     * The method processes the provided trajectory to compute a TD error, evaluates the update condition,
     * and modifies the Q-function accordingly. If the trajectory is empty or the computed TD error is zero,
     * the method returns the original Q-function without updates.
     *
     * @param Q The Q-function representing the state-action value function to be updated.
     *          It estimates the quality of specific actions in given states.
     * @param trajectory The trajectory consisting of a series of state-action transitions used to compute
     *                   the TD error and perform updates on the Q-function.
     * @return The updated Q-function if the TD error is non-zero, otherwise the original Q-function.
     */
    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        if (trajectory.isEmpty()) return Q

        val s0 = trajectory.first().state
        val a0 = trajectory.first().action

        val delta = td(Q, trajectory, policy, tailAction, gamma)
        if (delta == 0.0) return Q
        return Q.update(s0, a0, Q[s0, a0] + alpha() * delta)
    }
}
