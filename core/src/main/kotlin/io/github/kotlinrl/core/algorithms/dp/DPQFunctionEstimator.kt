package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

/**
 * Interface for defining estimators that update and refine Q-functions during
 * Dynamic Programming (DP) iterations in reinforcement learning.
 *
 * The `DPQFunctionEstimator` is used in scenarios where the Q-function, representing
 * the action-value function of a Markov Decision Process (MDP), needs to be
 * evaluated or updated based on a given probabilistic trajectory of states and actions.
 * This process is a core component of DP-based algorithms for solving MDPs.
 *
 * Implementations of this interface must define the method to estimate a new Q-function
 * based on the current Q-function and the provided trajectory, which typically contains
 * transition probabilities, rewards, and state-action pairs from the MDP model.
 *
 * @param State The type representing states in the environment or MDP.
 * @param Action The type representing possible actions that can be performed in the environment or MDP.
 */
interface DPQFunctionEstimator<State, Action> {
    /**
     * Estimates and updates the given Q-function based on a provided probabilistic trajectory
     * derived from a Markov Decision Process (MDP). The method refines the current
     * Q-function to better approximate the action-value function for the state-action pairs.
     *
     * @param Q the current `EnumerableQFunction` representing the action-value function
     *          of the MDP, which is to be updated.
     * @param trajectory a `ProbabilisticTrajectory` containing state-action sequences
     *                   and associated probabilities, rewards, and transitions used
     *                   for refining the Q-function.
     * @return an updated `EnumerableQFunction` that approximates the action-value function
     *         based on the provided trajectory.
     */
    fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableQFunction<State, Action>
}
