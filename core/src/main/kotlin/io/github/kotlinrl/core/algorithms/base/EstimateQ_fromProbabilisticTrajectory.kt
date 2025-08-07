package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * A functional interface used for estimating or updating a Q-function
 * based on a probabilistic trajectory. This approach leverages state-action
 * probabilities within the trajectory to compute the updated Q-function,
 * accounting for the probabilistic nature of the transitions.
 *
 * This interface is typically used in reinforcement learning scenarios
 * that involve stochastic environments or policies, where the trajectory
 * is represented with probabilistic transitions.
 *
 * It operates on an existing Q-function and computes a new Q-function
 * using the trajectory data, enabling iterative updates and learning.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions that can be performed.
 */
fun interface EstimateQ_fromProbabilisticTrajectory<State, Action> {
    /**
     * Estimates or updates a Q-function based on a probabilistic trajectory.
     * This method processes the given Q-function and the provided probabilistic trajectory
     * to compute a new Q-function that reflects updated estimations or learning outcomes.
     *
     * @param Q the current Q-function to be updated or used as a basis for the estimation.
     * @param trajectory the probabilistic trajectory consisting of state-action pairs
     *        and their associated probabilities, used for the Q-function's update or estimation.
     * @return the updated Q-function that incorporates information from the given trajectory.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): QFunction<State, Action>
}
