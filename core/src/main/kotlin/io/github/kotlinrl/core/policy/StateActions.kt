package io.github.kotlinrl.core.policy

/**
 * A functional interface representing a contract to determine the list of possible actions
 * for a given state in an environment.
 *
 * This serves as a utility in scenarios such as defining action spaces for policies
 * or algorithms in reinforcement learning, where knowing the set of feasible actions
 * for a specific state is essential.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions available in the environment.
 */
fun interface StateActions<State, Action> {
    /**
     * Determines the list of possible actions for a given state in the environment.
     *
     * @param state the current state of the environment for which actions are to be determined.
     * @return a list of actions that can be performed in the given state.
     */
    operator fun invoke(state: State): List<Action>
}