package io.github.kotlinrl.core.policy

interface QFunctionPolicy<State, Action> : Policy<State, Action> {
    val q: EnumerableQFunction<State, Action>
    val stateActions: StateActions<State, Action>

    fun probabilities(state: State): Map<Action, Double> {
        val actions = stateActions(state)
        val scores = actions.map { q[state, it] }
        val total = scores.sum()
        return actions.zip(scores.map { it / total }).toMap()
    }

    fun probability(state: State, action: Action): Double =
        q[state, action]
}