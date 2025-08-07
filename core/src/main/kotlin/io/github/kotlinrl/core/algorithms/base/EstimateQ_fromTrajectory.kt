package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * A functional interface designed to facilitate the estimation of
 * a new Q-function based on a given trajectory of state-action-reward transitions.
 *
 * Implementations of this interface define the algorithm or logic for updating
 * the Q-function, incorporating information from the trajectory to improve the
 * evaluation of state-action pairs. This can be useful in reinforcement learning
 * for processing complete episodes and refining policies.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 */
fun interface EstimateQ_fromTrajectory<State, Action> {
    /**
     * Estimates a new Q-function based on the provided trajectory of state-action transitions,
     * using the specified initial Q-function as a reference. This operator function applies
     * the logic defined in the implementation of the EstimateQ_fromTrajectory interface to
     * refine the Q-function for improved evaluation of state-action pairs.
     *
     * @param Q the initial Q-function used to evaluate the quality of state-action pairs.
     * @param trajectory the sequence of state-action-reward transitions representing an episode
     *        within the environment.
     * @return a Q-function representing the updated estimates of state-action pair values
     *         after processing the trajectory.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): QFunction<State, Action>
}