package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.EnumerableQFunction
import kotlin.random.*

/**
 * Represents a policy that selects actions randomly from the set of possible actions
 * for a given state. The policy assumes a uniform probability distribution over the
 * available actions, making the selection completely stochastic.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions available in the environment.
 * @property Q the Q-function representing the utility of state-action pairs.
 * @property stateActions a function that maps a state to the list of actions that can be taken in that state.
 * @property rng the random number generator used to select actions probabilistically.
 */
class RandomPolicy<State, Action>(
    override val Q: QFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val rng: Random = Random.Default
) : Policy<State, Action> {

    /**
     * Creates an improved policy based on the provided Q-function. This method returns a new policy
     * instance that stochastically selects actions using a uniform probability distribution
     * across all available actions for each state, determined by the given Q-function.
     *
     * @param Q the Q-function representing the utility of state-action pairs, used as a reference
     * for creating the new improved policy.
     * @return a new policy instance that uses the provided Q-function and selects actions randomly
     * from the available actions for each state.
     */
    override fun improve(Q: QFunction<State, Action>): Policy<State, Action> =
        RandomPolicy(Q, stateActions, rng)

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
