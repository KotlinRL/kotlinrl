package io.github.kotlinrl.core.policy

import kotlin.random.*

class UniformStochasticPolicy<State, Action>(
    override val q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    rng: Random = Random.Default
) : StochasticPolicy<State, Action>(rng) {
    private val randomPolicy = RandomPolicy(stateActions, rng)

    override fun invoke(state: State): Action {
        return randomPolicy(state)
    }

    override fun improve(q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        UniformStochasticPolicy(q, stateActions, rng)

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActions(state)
        val p = 1.0 / actions.size
        return actions.map { it to p }
    }
}
