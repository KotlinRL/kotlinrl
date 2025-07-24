package io.github.kotlinrl.core.policy

import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy<State, Action>(
    override val qTable: QFunction<State, Action>,
    private val temperature: ParameterSchedule,
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random
) : StochasticPolicy<State, Action>(stateActionListProvider, rng), QFunctionPolicy<State, Action> {

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