package io.github.kotlinrl.core.api

import io.github.kotlinrl.core.*

/**
 * Represents a decision-making policy in reinforcement learning or similar frameworks
 * that maps a given state to an action or a probability distribution over possible actions.
 *
 * This interface provides two primary operations:
 * - Deterministic action selection using the `invoke` operator function.
 * - Probabilistic action selection using the `get` operator function to fetch a distribution.
 *
 * @param State The type representing the states over which the policy operates.
 * @param Action The type representing the actions determined by the policy.
 */
interface Policy<State, Action> {
    /**
     * Determines the action to be taken for the provided state based on the policy.
     *
     * This operator function allows for invoking the policy directly with a state
     * to compute and return the corresponding action.
     *
     * @param state the current state of the environment for which an action is to be decided.
     * @return the action determined by the policy for the given state.
     */
    operator fun invoke(state: State): Action
    /**
     * Retrieves the probability distribution over actions for the given state based on the policy.
     *
     * This operator function allows fetching a `Distribution` object that represents the likelihood
     * of each possible action in the provided state. The `Distribution` can subsequently be used for
     * probabilistic action selection or other operations like sampling or computing log-probabilities.
     *
     * @param state the current state for which the probability distribution over actions is required.
     * @return the probability distribution over actions for the given state.
     */
    operator fun get(state: State): Distribution<Action>
}