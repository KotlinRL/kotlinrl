package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

/**
 * Defines a functional interface for evaluating whether a given transition
 * marks a successful termination condition in a reinforcement learning episode.
 *
 * This interface is used to encapsulate the logic for determining success in the context
 * of the training process. It enables evaluation of environment-specific criteria
 * through a user-defined implementation.
 *
 * Generic parameters:
 * @param State The type representing the state in the environment.
 * @param Action The type representing the action taken by the agent.
 *
 * Functional Method:
 * @param result A `Transition` object encapsulating the details of the current
 *               state, action, reward, next state, and termination information.
 * @return A boolean indicating whether the specified `result` meets the criteria
 *         for successful termination.
 */
fun interface SuccessfulTermination<State, Action> {
    /**
     * Evaluates whether a given transition marks a successful termination condition
     * in the context of a reinforcement learning episode.
     *
     * This function is invoked with a `Transition` object and determines whether the
     * specified transition meets the criteria for success.
     *
     * @param result A `Transition` object encapsulating the current state, action, reward,
     *               resulting next state, and termination or truncation information.
     * @return A boolean value indicating whether the provided transition satisfies the
     *         successful termination condition.
     */
    operator fun invoke(result: Transition<State, Action>): Boolean
}