package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Interface representing a Q-function estimator based on observed state transitions in reinforcement learning algorithms.
 *
 * This estimator provides a mechanism to update an existing Q-function based on a given transition by deriving
 * a new Q-function that incorporates the information from the transition. It is typically used to improve the
 * value function in iterative learning methods.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 */
interface TransitionQFunctionEstimator<State, Action> {
    /**
     * Estimates a new Q-function by applying state transition information to an existing Q-function.
     * This method is used for updating Q-functions based on observed transitions in reinforcement learning.
     *
     * @param Q the existing Q-function to be updated, which provides expected values for state-action pairs.
     * @param transition the observed transition containing the current state, the action performed,
     * the reward received, and the resulting next state.
     * @return the updated Q-function that incorporates the information from the provided transition.
     */
    fun estimate(Q: EnumerableQFunction<State, Action>, transition: Transition<State, Action>): EnumerableQFunction<State, Action>
}