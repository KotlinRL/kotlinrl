package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.*

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
     * Simulates the outcome of performing a specific action in a given state without altering
     * the environment's actual state. This function executes a hypothetical step within
     * the environment to predict the resulting state, reward, and termination status.
     *
     * @param state The initial state within the environment where the action is to be simulated.
     * @param action The action to be simulated in the provided state.
     * @return A StepResult containing the resulting state, the reward obtained from the simulation,
     * the termination status (whether the environment would conclude), the truncation status,
     * and additional auxiliary information about the simulation outcome.
     */
    fun simulateStep(state: State, action: Action): StepResult<State>
}