package io.github.kotlinrl.core.policy

/**
 * Represents an abstraction for decision-making strategies in a reinforcement learning environment.
 * A policy defines the rules and mechanisms for selecting actions based on the current state,
 * aiming to optimize the performance of an agent.
 *
 * @param State the type representing the state of the environment.
 * @param Action the type representing the actions available to the agent.
 */
interface Policy<State, Action> {
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
    fun improve(Q: QFunction<State, Action>): Policy<State, Action>

    /**
     * Represents the Q-function used by the policy for determining the quality of state-action pairs.
     *
     * The Q-function evaluates the expected cumulative reward for given state-action pairs,
     * allowing the policy to make informed decisions based on the quality of actions
     * in specific states. This property is essential for reinforcement learning policies
     * to decide on actions that maximize the agent's performance in the environment.
     *
     * The Q-function is enumerable, meaning it can explicitly iterate over all possible
     * states, making it suitable for environments with a finite and manageable state space.
     */
    val Q: QFunction<State, Action>

    /**
     * Represents a function that maps a given state to a list of available actions.
     *
     * This abstraction is used in reinforcement learning to determine the possible
     * actions that an agent can take given a specific state in the environment. It acts
     * as a utility for state-action evaluation and action selection in various policies.
     *
     * @param State the type of the state in the environment.
     * @param Action the type of the actions available in the environment.
     */
    val stateActions: StateActions<State, Action>

    /**
     * Computes the probabilities of selecting each possible action in the given state based on the Q-values.
     *
     * @param state the current state for which action probabilities are to be calculated.
     * @return a map where keys are actions and values are the probabilities of selecting each action.
     */
    fun probabilities(state: State): Map<Action, Double> {
        val actions = stateActions(state)
        val scores = actions.map { Q[state, it] }
        val total = scores.sum()
        return actions.zip(scores.map { it / total }).toMap()
    }

    /**
     * Computes the probability of selecting a specific action for a given state based on the Q-values.
     *
     * @param state the current state for which the action probability is to be determined.
     * @param action the action whose selection probability is to be calculated.
     * @return the probability of selecting the specified action in the given state.
     */
    fun probability(state: State, action: Action): Double =
        Q[state, action]
}