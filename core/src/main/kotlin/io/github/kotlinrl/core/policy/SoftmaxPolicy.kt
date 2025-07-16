package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.ExplorationFactor
import io.github.kotlinrl.core.algorithms.*
import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val temperature: ExplorationFactor,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random
) : ProbabilisticPolicy<State, Action>(rng) {

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val temperature = temperature()
        val actions = stateActionListProvider(state)
        return actions.map { action ->
            val q = qTable[state, action]
            val score = exp(q / temperature)
            action to score
        }
    }
}