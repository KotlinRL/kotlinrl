package io.github.kotlinrl.core.policy

import kotlin.random.*

/**
 * A stochastic policy implementation that selects actions randomly based on the defined action space
 * for a given state. It is often used as a baseline or exploratory policy in reinforcement learning.
 *
 * This policy assumes all actions are equally likely and chooses one uniformly at random.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 * @param stateActions a function providing the list of available actions for a given state.
 * @param rng a random number generator used to select actions uniformly at random.
 */
class RandomPolicy<State, Action>(
    private val stateActions: StateActions<State, Action>,
    private val rng: Random = Random.Default
) : Policy<State, Action> {

    /**
     * Selects an action for the given state by randomly choosing from the list of available actions
     * provided by the stateActions function. The selection is stochastic and assumes a uniform
     * probability distribution across all actions.
     *
     * @param state the current state of the environment for which an action is to be selected.
     * @return the selected action for the given state.
     */
    override operator fun invoke(state: State): Action {
        val actions = stateActions(state)
        return actions[rng.nextInt(actions.size)]
    }
}
