package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.policy.EnumerableValueFunction

/**
 * Represents an interface for defining value function estimators in the context of
 * Dynamic Programming (DP) techniques for reinforcement learning.
 *
 * The `DPValueFunctionEstimator` is utilized to update and refine state-value functions
 * (V-functions) based on trajectories of states and transitions derived from an environment
 * or Markov Decision Process (MDP). This process is fundamental to DP approaches for solving
 * MDPs, where the value of each state is iteratively improved to better approximate the
 * optimal value function.
 *
 * Implementations of this interface must provide a method to estimate and update the value
 * function using a provided trajectory. The trajectory typically contains information
 * about state transitions, rewards, and probabilities, which are used to compute improved
 * state-value estimates.
 *
 * @param State The type representing states in the environment or MDP.
 * @param Action The type representing actions that can be performed in the environment or MDP.
 */
interface DPValueFunctionEstimator<State, Action> {
    /**
     * Estimates and updates the provided state-value function based on the given probabilistic trajectory.
     * This method evaluates the current value function and updates its estimates using the transitions,
     * probabilities, and rewards described in the trajectory.
     *
     * @param V the existing `EnumerableValueFunction` representing the state-value function
     *          to be updated.
     * @param trajectory a `ProbabilisticTrajectory` containing sequences of states,
     *                   probabilities, and rewards used to refine the state-value function.
     * @return an updated `EnumerableValueFunction` that represents an improved approximation
     *         of the state-value function according to the trajectory.
     */
    fun estimate(
        V: EnumerableValueFunction<State>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableValueFunction<State>
}
