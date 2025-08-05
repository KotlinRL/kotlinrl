package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

/**
 * Represents a Markov Decision Process (MDP) model. It provides the necessary
 * methods to define the states, actions, transitions, and rewards within the MDP.
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 */
interface MDPModel<State, Action> {
    /**
     * Retrieves a list of all states in the Markov Decision Process (MDP) model.
     *
     * @return A list containing all states defined in the MDP.
     */
    fun allStates(): List<State>

    /**
     * Retrieves a list of all possible actions in the Markov Decision Process (MDP) model.
     *
     * @return A list containing all actions defined in the MDP.
     */
    fun allActions(): List<Action>

    /**
     * Retrieves the probabilistic transitions for a given state-action pair in the Markov Decision Process (MDP).
     *
     * @param state The current state in the MDP.
     * @param action The action taken from the given state.
     * @return A probabilistic trajectory consisting of possible next states, associated actions, and their probabilities.
     */
    fun transitions(state: State, action: Action): ProbabilisticTrajectory<State, Action>

    /**
     * Computes the expected reward for a given state and action in the Markov Decision Process (MDP).
     *
     * @param state The current state in the MDP.
     * @param action The action taken from the given state.
     * @return The expected reward obtained by taking the specified action from the given state.
     */
    fun expectedReward(state: State, action: Action): Double
}
