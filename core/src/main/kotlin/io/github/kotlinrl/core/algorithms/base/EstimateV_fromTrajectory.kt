package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents a functional interface for estimating a value function (V) based on a given trajectory.
 *
 * This interface defines a single method that acts as an operator function, allowing the estimation
 * of a new value function derived from the provided trajectory. The method takes an initial value
 * function and a trajectory of states and actions, and returns the updated value function
 * that reflects the learning or adaptation based on the sequence of transitions.
 *
 * Designed to be a flexible and extensible structure, this interface can be implemented to perform
 * various forms of value function estimation, such as Monte Carlo methods, Temporal Difference (TD) learning,
 * or other trajectory-based approaches in reinforcement learning.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 */
fun interface EstimateV_fromTrajectory<State, Action> {
    /**
     * Estimates a new value function based on the given trajectory and initial value function.
     *
     * This operator function computes the updated `ValueFunction` by processing the provided
     * trajectory of states and actions, starting with an initial estimation of the value function.
     *
     * @param V the initial value function to be updated based on the trajectory.
     * @param trajectory the sequence of states and actions representing the trajectory to be used
     *        for updating the value function.
     * @return the updated value function estimated from the trajectory.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        trajectory: Trajectory<State, Action>
    ): ValueFunction<State>
}
