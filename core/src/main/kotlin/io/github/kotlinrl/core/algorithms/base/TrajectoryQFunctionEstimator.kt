package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Interface for estimating Q-functions based on trajectory data.
 *
 * A `TrajectoryQFunctionEstimator` provides a mechanism to derive an updated Q-function
 * from a given trajectory of state-action-reward transitions. The estimation is performed
 * based on an initial Q-function and a trajectory representing the sequence of transitions
 * an agent experienced during an episode of interaction with its environment.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 */
interface TrajectoryQFunctionEstimator<State, Action> {
    /**
     * Estimates and returns an updated Q-function based on the given initial Q-function and trajectory data.
     * This method derives the updated Q-function by processing the sequence of state-action-reward transitions
     * in the trajectory and using the provided initial Q-function as a reference.
     *
     * @param Q the initial enumerable Q-function that serves as the baseline for updates.
     * @param trajectory the sequence of state-action-reward transitions observed during interaction
     *        with the environment.
     * @return an updated enumerable Q-function derived from the trajectory and initial Q-function.
     */
    fun estimate(Q: EnumerableQFunction<State, Action>, trajectory: Trajectory<State, Action>): EnumerableQFunction<State, Action>
}