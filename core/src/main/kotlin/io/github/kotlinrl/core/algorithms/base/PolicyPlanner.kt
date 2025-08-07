package io.github.kotlinrl.core.algorithms.base

import io.github.kotlinrl.core.*

/**
 * Interface defining a contract to construct policies based on Q-functions, environment models,
 * and state-action mappings for Markov Decision Processes (MDP).
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the possible actions within the environment.
 */
interface PolicyPlanner<State, Action> {
    /**
     * Constructs a policy using the provided Q-function, model, and state-action mappings.
     *
     * @param Q the Q-function that evaluates the quality of state-action pairs, guiding decision-making.
     * @param model the Markov Decision Process (MDP) model representing the environment's dynamics.
     * @param stateActions the set of allowable actions for each state in the environment.
     * @return a policy mapping states to actions based on the provided inputs.
     */
    operator fun invoke(
        Q: QFunction<State, Action>,
        model: MDPModel<State, Action>,
        stateActions: StateActions<State, Action>
    ): Policy<State, Action>
}