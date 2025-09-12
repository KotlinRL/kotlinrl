package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.model.*
import io.github.kotlinrl.core.space.*

/**
 * A specialized type of environment where the transition dynamics are explicitly defined
 * and can be simulated given a specific state and action. This interface extends the generic
 * environment model by providing additional functionality to predict or simulate the outcome
 * of performing an action in a given state without directly modifying the environment's state.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing the actions that can be performed in the environment.
 * @param ObservationSpace The type of space defining the structure and constraints of the states.
 * @param ActionSpace The type of space defining the structure and constraints of actions.
 */
interface ModelBasedEnv<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>
    : Env<State, Action, ObservationSpace, ActionSpace> {

    /**
     * Converts the current model-based environment into a Markov Decision Process (MDP).
     *
     * This function provides a mathematical representation of the environment by returning
     * an equivalent MDP instance. The MDP encapsulates states, actions, transition probabilities,
     * rewards, and a discount factor necessary for formal decision-making analysis.
     *
     * @return An `MDP` instance representing the Markov Decision Process derived from the environment.
     */
    fun asMDP(gamma: Double = 0.9): MDP<State, Action>
}