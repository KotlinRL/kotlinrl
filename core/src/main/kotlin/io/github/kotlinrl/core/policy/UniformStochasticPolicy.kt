package io.github.kotlinrl.core.policy

import kotlin.random.*

class UniformStochasticPolicy<State, Action>(
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
) : StochasticPolicy<State, Action>(stateActionListProvider, rng), PolicyImprovementStrategy<State, Action> {
    private val randomPolicy = RandomPolicy(stateActionListProvider, rng)

    override fun invoke(state: State): Action {
        return randomPolicy(state)
    }

    override operator fun invoke(q: QFunction<State, Action>): Policy<State, Action> = this

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActionListProvider(state)
        val p = 1.0 / actions.size
        return actions.map { it to p }
    }
}
