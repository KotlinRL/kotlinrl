package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * A functional interface that defines a strategy for estimating an updated value function
 * based on a given value function and a state-action transition.
 *
 * This interface is intended for use in reinforcement learning algorithms, where the
 * value function plays a critical role in evaluating the expected rewards of states,
 * and transitions represent the dynamics of the environment. Through this interface,
 * the value function can be recalculated or adjusted based on observed transitions.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing actions that can be performed within the environment.
 */
fun interface EstimateV_fromTransition<State, Action> {
    /**
     * Updates the value function based on a given state-action transition and an existing value function.
     *
     * This operator function is intended to compute a new value function by applying
     * an update strategy to the provided transition and value function. The result
     * is typically used to guide learning algorithms in reinforcement learning frameworks.
     *
     * @param V the current value function representing state value estimates.
     * @param transition the state-action transition, including current state, action, next state,
     *        and the reward observed from the environment.
     * @return the updated value function after applying the transition.
     */
    operator fun invoke(
        V: ValueFunction<State>,
        transition: Transition<State, Action>
    ): ValueFunction<State>
}