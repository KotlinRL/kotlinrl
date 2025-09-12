package io.github.kotlinrl.core.model

/**
 * Represents a finite tabular Markov Decision Process (MDP) where states and actions are integer-encoded
 * and finite in number. This interface provides structure for working with MDPs with predefined, enumerable
 * states and actions.
 *
 * @property S A finite set of integer states, defining the state space of the MDP.
 * @property A A fixed set of integer actions, representing the action space of the MDP.
 * @property numStates The total number of states in the MDP, derived from the finite state set.
 */
interface FiniteTabular : MDP<Int, Int> {
    /**
     * Represents the finite set of states for the Markov Decision Process (MDP).
     *
     * This property defines the state space of the MDP, where the states are integer-encoded
     * and enumerable. It provides a structured way to represent all possible states
     * within a decision-making or reinforcement learning environment.
     *
     * @see FiniteStates
     * @see io.github.kotlinrl.core.model.TabularMDP
     */
    override val S: FiniteStates

    /**
     * Represents the fixed set of integer-encoded actions available in a finite tabular
     * Markov Decision Process (MDP). This property defines the complete action space
     * of the MDP, where the number of possible actions is determined by the implementation
     * of `FixedIntActions`.
     *
     * The action space is uniform across all states, and each action is defined as an integer
     * index ranging from 0 to `numActions - 1`. This ensures a consistent and constrained
     * representation of actions within the decision-making framework of the MDP.
     *
     * @see FixedIntActions
     */
    override val A: FixedIntActions

    /**
     * Returns the total number of actions available for a given integer-encoded state in a finite
     * tabular Markov Decision Process (MDP).
     *
     * The number of actions is constant and predefined for all states in the MDP, as defined
     * by the `FixedIntActions` implementation.
     *
     * @param state The integer-encoded state for which the number of actions is queried.
     * @return The total count of available actions for the specified state.
     */
    fun numActions(state: Int): Int = A.numActions

    /**
     * Retrieves the total number of states in a finite Markov Decision Process (MDP).
     *
     * This property provides access to the size of the state set, which is non-null for finite
     * state representations. It helps in determining the overall scope of the state space
     * in environments such as reinforcement learning or decision-making tasks.
     *
     * @return The total count of states as defined by the size of the finite state set.
     */
    val numStates: Int
        get() = S.sizeOrNull
}