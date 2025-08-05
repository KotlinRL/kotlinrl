package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

/**
 * Represents a learnable Markov Decision Process (MDP) model, extending the functionality of a basic `MDPModel`.
 * This interface provides additional capabilities to update state transition knowledge, sample transitions,
 * and verify whether specific state-action pairs are known within the model.
 *
 * @param State The type representing the states in the MDP.
 * @param Action The type representing the actions in the MDP.
 */
interface LearnableMDPModel<State, Action> : MDPModel<State, Action> {
    /**
     * Updates the model with the given state transition information. This method incorporates the specified
     * transition data, including the state, action, resulting reward, and next state, into the model's internal
     * representation. It can be used to update the knowledge of state transitions and the associated rewards.
     *
     * @param transition The transition to be recorded, including the current state, the action taken, the reward received,
     * and the resulting next state.
     */
    fun update(transition: Transition<State, Action>)
    /**
     * Samples a transition from the model's current state-action transition knowledge.
     * The method attempts to randomly select a known state-action pair and one of its possible outcomes
     * based on the recorded transitions and their probabilities in the model. If no transitions are known,
     * it will return null.
     *
     * @return A sampled transition encapsulated as a `Transition<State, Action>` object, or null if no transitions are available.
     */
    fun sampleTransition(): Transition<State, Action>?
    /**
     * Determines whether the state-action pair is known to the model.
     * A state-action pair is considered known if it has been observed and recorded in the model's internal state.
     *
     * @param state The state in the state-action pair to be checked.
     * @param action The action in the state-action pair to be checked.
     * @return True if the state-action pair is known to the model, false otherwise.
     */
    fun isKnown(state: State, action: Action): Boolean
}
