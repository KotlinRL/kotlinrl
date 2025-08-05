package io.github.kotlinrl.core.policy

/**
 * A functional interface representing a policy in reinforcement learning.
 *
 * A policy defines the behavior of an agent, determining the action to be taken
 * based on the current state of the environment. This interface describes the
 * essential contract of a policy, allowing deterministic or stochastic decision-making
 * strategies to be implemented.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed.
 */
fun interface Policy<State, Action> {
    /**
     * Determines the action to be taken based on the given state.
     *
     * @param state the current state of the environment.
     * @return the action to be performed for the given state.
     */
    operator fun invoke(state: State): Action

    /**
     * Creates an improved policy based on the given Q-function. The improved policy is derived by
     * leveraging the provided Q-function to optimize the decision-making process for an agent.
     *
     * @param Q the Q-function representing the expected cumulative reward for each state-action pair.
     * @return the improved policy that uses the provided Q-function for optimized decision-making.
     */
    fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> = this
}