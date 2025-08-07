package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Functional interface for estimating an updated Q-function from a given transition.
 *
 * This interface represents a function responsible for modifying or recalculating
 * the Q-function based on a single state-action-reward-next-state transition. The goal
 * is to update the Q-function to better approximate the quality of state-action pairs
 * as reinforcement learning progresses.
 *
 * @param State the type representing the states of the environment.
 * @param Action the type representing the actions performable in the environment.
 */
fun interface EstimateQ_fromTransition<State, Action> {
    /**
     * Updates the given Q-function based on the provided transition and returns the updated Q-function.
     *
     * This function accepts an existing Q-function and a state-action-reward-next-state transition.
     * It applies logic to modify or update the Q-function to better reflect the consequences of
     * the given transition in a reinforcement learning context.
     *
     * @param Q the current Q-function representing state-action value estimates.
     * @param transition the state-action-reward-next-state transition used to update the Q-function.
     * @return the updated Q-function incorporating the effects of the provided transition.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        transition: Transition<State, Action>
    ): QFunction<State, Action>
}