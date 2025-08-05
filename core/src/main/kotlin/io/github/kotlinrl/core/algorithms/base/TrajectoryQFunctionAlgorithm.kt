package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents an abstract reinforcement learning algorithm based on trajectory-driven Q-function estimation.
 * This algorithm leverages trajectory data to progressively estimate and improve the Q-function and the policy.
 * It extends the `QFunctionAlgorithm` to incorporate trajectory-based Q-function updates and policy improvements.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial Q-function policy used by the algorithm.
 * @param estimator the estimator used to derive Q-function updates from observed trajectories.
 * @param onPolicyUpdate a callback function invoked when the policy is updated.
 * @param onQFunctionUpdate a callback function invoked when the Q-function is updated.
 */
abstract class TrajectoryQFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    estimator: TrajectoryQFunctionEstimator<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { }
) : QFunctionAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    /**
     * Represents a component responsible for generating Q-function predictions using a trajectory-driven
     * approach. The predictions are derived from an initial Q-function and updated based on updates provided
     * by a trajectory-based Q-function estimator.
     *
     * The `prediction` property is used within the trajectory-based reinforcement learning algorithm to
     * process trajectories of transitions. It leverages the concrete implementation of
     * `TrajectoryQFunctionEstimator` to estimate updates to the current Q-function, which in turn influences
     * the policy improvements in the algorithm.
     *
     * This property is initialized with an initial Q-function (`Q`) and a trajectory-based Q-function
     * estimator (`estimator`) and remains accessible to subclasses while encapsulating the internal updates.
     */
    protected val prediction = TrajectoryQFunctionPrediction(Q, estimator)

    /**
     * Observes and processes the provided trajectory and episode data to update the Q-function and policy.
     *
     * This method uses the given trajectory to update the Q-function values via the `prediction` component
     * and then computes an improved policy based on the updated Q-function. It delegates the update of
     * the Q-function to the associated `prediction` mechanism and employs policy improvement logic
     * to refine the decision-making process.
     *
     * @param trajectory The sequence of state-action-reward transitions observed during an episode.
     *                   It represents the path followed by an agent within the environment.
     * @param episode The index or identifier of the current episode associated with the given trajectory.
     */
    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        prediction(trajectory, episode)
        Q = prediction.Q
        policy = policy.improve(Q)
    }
}