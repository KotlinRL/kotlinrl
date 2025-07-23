package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.ExplorationFactor
import kotlin.random.*

class EpsilonSoftPolicy<State, Action>(
    override val qTable: QFunction<State, Action>,
    private val explorationFactor: ExplorationFactor,
   stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random
) : StochasticPolicy<State, Action>(stateActionListProvider, rng), QFunctionPolicy<State, Action> {

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActionListProvider(state)
        val greedyAction = qTable.bestAction(state)
        val n = actions.size
        val epsilon = explorationFactor()

        return actions.map { action ->
            val prob = if (action == greedyAction) {
                (1 - epsilon) + (epsilon / n)
            } else {
                epsilon / n
            }
            action to prob
        }
    }
}