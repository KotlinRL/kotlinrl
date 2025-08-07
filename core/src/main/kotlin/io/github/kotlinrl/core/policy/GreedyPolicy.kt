package io.github.kotlinrl.core.policy

/**
 * A greedy policy implementation in reinforcement learning that selects actions based
 * solely on the Q-function's evaluation. The policy chooses the action with the highest
 * Q-value for a given state, thus maximizing the expected reward deterministically.
 *
 * This policy is considered "greedy" because it always selects the action with the best
 * immediate Q-value, which may not account for long-term exploration or potential future rewards.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions available in the environment.
 * @property Q the Q-function used to evaluate state-action pairs and determine the best action.
 * @property stateActions the utility function to determine the possible actions for a given state.
 */
class GreedyPolicy<State, Action>(
    override val Q: QFunction<State, Action>,
    override val stateActions: StateActions<State, Action>
) : Policy<State, Action> {

    /**
     * Determines the action to be taken for a given state based on the Q-function.
     *
     * This method invokes the Q-function to select the action that maximizes the
     * Q-value for the given state. It represents a greedy policy where the best action
     * is selected deterministically, based purely on the Q-function's evaluation.
     *
     * @param state the state for which the action is to be determined.
     * @return the action that corresponds to the highest Q-value for the given state.
     */
    override operator fun invoke(state: State): Action {
        return Q.bestAction(state)
    }

    /**
     * Creates an improved policy based on the provided Q-function. The improved policy
     * is derived by utilizing the Q-function to optimize decision-making, typically
     * selecting actions that maximize expected rewards.
     *
     * @param Q the Q-function representing the expected cumulative rewards for each state-action pair.
     * @return the improved policy constructed using the provided Q-function.
     */
    override fun improve(Q: QFunction<State, Action>): Policy<State, Action> =
        GreedyPolicy(Q, stateActions)
}