package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Class responsible for updating a value function using trajectory data in reinforcement learning.
 *
 * This class processes observed trajectories and updates the enumerable value function
 * based on the information provided by the `TrajectoryValueFunctionEstimator`.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 * @param initialV The initial enumerable value function to be updated across episodes.
 * @param estimator The estimator that defines how the value function is updated using trajectory information.
 */
class TrajectoryValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val estimator: TrajectoryValueFunctionEstimator<State, Action>,
) : TrajectoryObserver<State, Action> {

    /**
     * Represents the enumerable value function being updated during the prediction process.
     *
     * This variable holds the current state of the value function, which is updated across
     * episodes using trajectory data. The updates are performed by the associated
     * `TrajectoryValueFunctionEstimator` based on the observed trajectories.
     *
     * The value function maps states to their estimated values, aiding in reinforcement learning
     * tasks. The property is initialized with an initial value function and is updated only
     * internally within this class.
     */
    var V = initialV
        private set

    /**
     * Processes a trajectory and updates the value function for a specific episode.
     *
     * This method takes a trajectory consisting of state-action transitions and updates the enumerable value function
     * `V` using the provided `TrajectoryValueFunctionEstimator`. The trajectory provides information regarding the sequence
     * of states and actions, which is used to compute the updated value function for the current episode.
     *
     * @param trajectory the sequence of state-action transitions to update the value function.
     * @param episode the index or identifier of the current episode being processed.
     */
    override operator fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        V = estimator.estimate(V, trajectory)
    }
}