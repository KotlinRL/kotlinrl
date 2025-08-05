package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.Trajectory
import io.github.kotlinrl.core.TrajectoryObserver

/**
 * Represents a class responsible for updating an enumerable Q-function based on trajectory
 * data using a specified `TrajectoryQFunctionEstimator`. This class observes trajectories
 * and applies estimation to derive updated Q-function values, reflecting the improved
 * understanding of action-state value relationships.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialQ the initial enumerable Q-function, serving as the baseline for updates.
 * @param estimator the `TrajectoryQFunctionEstimator` used to estimate and derive updated
 * Q-function values from observed trajectories.
 */
class TrajectoryQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: TrajectoryQFunctionEstimator<State, Action>,
) : TrajectoryObserver<State, Action> {

    /**
     * Represents the enumerable Q-function used to estimate the value of taking certain
     * actions in specific states within the environment. This property is updated
     * iteratively, using a `TrajectoryQFunctionEstimator` to reflect the improved understanding
     * of action-state value relationships based on observed trajectories.
     *
     * The Q-function serves as the core representation of the expected rewards for
     * actions in given states and is essential for decision-making in reinforcement learning.
     *
     * This property is initialized with the provided `initialQ` and can only be updated
     * internally within the class.
     */
    var Q = initialQ
        private set

    /**
     * Updates the enumerable Q-function based on the provided trajectory and episode index.
     * This method applies a `TrajectoryQFunctionEstimator` to derive an updated Q-function
     * by processing the sequence of state-action-reward transitions within the trajectory.
     *
     * @param trajectory the sequence of state-action-reward transitions observed during an interaction
     *        with the environment, representing the data from which the Q-function will be updated.
     * @param episode the index or identifier of the episode corresponding to the provided trajectory.
     */
    override operator fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        Q = estimator.estimate(Q, trajectory)
    }
}