package io.github.kotlinrl.core.model

/**
 * Represents a probabilistic transition in a Markov Decision Process (MDP).
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 * @property state The current state of the MDP.
 * @property action The action taken from the current state.
 * @property reward The reward obtained after taking the action from the current state.
 * @property nextState The state resulting from the transition after taking the action.
 * @property probability The probability of transitioning to the next state when taking the action.
 * @property done Indicates whether the transition results in a terminal state.
 */
data class ProbabilisticTransition<State, Action>(
    val state: State,
    val action: Action,
    val reward: Double,
    val nextState: State,
    val probability: Double,
    val done: Boolean
)