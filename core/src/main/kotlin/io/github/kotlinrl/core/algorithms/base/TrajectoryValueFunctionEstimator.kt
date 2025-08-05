package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Defines an interface for estimating value functions using trajectory data.
 *
 * This interface represents a contract for creating implementations that can take a given
 * value function and a trajectory of state-action transitions to produce a new, updated
 * value function based on the information provided in the trajectory.
 *
 * @param State the type representing the environment's states.
 * @param Action the type representing the actions that can be executed in the environment.
 */
interface TrajectoryValueFunctionEstimator<State, Action> {
    /**
     * Estimates a new value function based on a given initial value function and a trajectory of state-action transitions.
     *
     * This method takes an initial enumerable value function and processes a trajectory to produce a new value function
     * that reflects the information contained within the trajectory.
     *
     * @param V the initial enumerable value function to be updated or transformed.
     * @param trajectory the sequence of state-action transitions to be used for value function estimation.
     * @return the newly estimated enumerable value function updated based on the provided trajectory.
     */
    fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State>
}
