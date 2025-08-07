package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*
import kotlin.math.*

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

    /**
     * Computes the probabilistic trajectory for all state-action pairs based on the given state-actions mapping.
     *
     * @param stateActions A mapping function that provides a list of possible actions for each state.
     * @return A probabilistic trajectory represented as a combination of states, actions, and associated transitions.
     */
    fun probabilisticTrajectory(
        stateActions: StateActions<State, Action>
    ): ProbabilisticTrajectory<State, Action> = allStates().flatMap { s ->
        stateActions(s).flatMap { a ->
            transitions(s, a)
        }
    }

    /**
     * Computes the probabilistic trajectory for all states in the Markov Decision Process (MDP)
     * based on the given policy.
     *
     * @param policy A policy that maps each state to a specific action to be taken.
     * @return A probabilistic trajectory consisting of states, actions, and their associated transitions for all states.
     */
    fun probabilisticTrajectory(
        policy: Policy<State, Action>
    ): ProbabilisticTrajectory<State, Action> = allStates().flatMap { state ->
        transitions(state, policy(state))
    }

    /**
     * Computes the maximum absolute difference between two Q-functions over all state-action pairs.
     *
     * @param stateActions A function mapping each state to a list of possible actions.
     * @param currentQ The current Q-function representing state-action values.
     * @param newQ The updated Q-function to be compared against the current Q-function.
     * @return The maximum absolute difference between the values of the given Q-functions for all state-action pairs.
     */
    fun deltaQ(
        stateActions: StateActions<State, Action>,
        currentQ: QFunction<State, Action>,
        newQ: QFunction<State, Action>
    ): Double = allStates().flatMap { state ->
        stateActions(state).map { action ->
            abs(currentQ[state, action] - newQ[state, action])
        }
    }.maxOrNull() ?: 0.0

    /**
     * Computes the maximum absolute difference between two value functions over all states.
     *
     * @param stateActions A function mapping each state to a list of possible actions.
     * @param currentV The current value function representing state values.
     * @param newV The updated value function to be compared against the current value function.
     * @return The maximum absolute difference between the values of the given value functions for all states.
     */
    fun deltaV(
        stateActions: StateActions<State, Action>,
        currentV: ValueFunction<State>,
        newV: ValueFunction<State>
    ): Double = allStates().maxOf {  abs(currentV[it] - newV[it]) }
}
