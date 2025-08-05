package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.Policy

/**
 * Represents a functional interface for creating policies in reinforcement learning.
 *
 * A `Planner` is responsible for generating a decision-making policy based on the
 * current state of the environment and the actions available. This interface
 * abstracts the process of policy creation to allow different planning strategies
 * to be implemented.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 */
fun interface Planner<State, Action> {
    /**
     * Invokes the planner to create a decision-making policy for reinforcement learning.
     *
     * @return a policy that maps states to actions, guiding the decision-making process.
     */
    operator fun invoke(): Policy<State, Action>
}