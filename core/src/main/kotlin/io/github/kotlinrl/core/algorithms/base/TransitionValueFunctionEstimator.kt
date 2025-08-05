package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Represents an interface for estimating value functions based on observed transitions in reinforcement learning.
 * Implementations of this interface define methods for generating an updated value function by utilizing
 * a given value function and a single transition.
 *
 * This interface serves as a foundational component in scenarios where dynamic estimation of value functions
 * is required, particularly within the context of reinforcement learning algorithms that involve iterative
 * updates of value representations.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 */
interface TransitionValueFunctionEstimator<State, Action> {
    /**
     * Estimates a new enumerable value function based on the given current value function and a single transition.
     *
     * This method takes an existing `EnumerableValueFunction` and a specific `Transition` instance, applies
     * a reinforcement learning estimation process, and returns an updated `EnumerableValueFunction` reflecting
     * the changes induced by the observed transition.
     *
     * @param V the current enumerable value function to be updated.
     * @param transition the observed transition containing the initial state, the action taken, the resulting state,
     * and the reward received.
     * @return the updated enumerable value function after incorporating the effects of the observed transition.
     */
    fun estimate(V: EnumerableValueFunction<State>, transition: Transition<State, Action>): EnumerableValueFunction<State>
}