package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy<State, Action>(
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val qTable: QTable<State, Action>,
    private val temperature: ExplorationFactor,
    rng: Random = Random.Default
) : ProbabilisticPolicy<State, Action>(rng) {

    override fun invoke(state: State): Action {
        val availableActions = stateActionListProvider(state)
        val T = temperature()
        val qValues = availableActions.map { qTable[state, it] }
        val maxQ = qValues.maxOrNull() ?: 0.0
        val scaled = qValues.map { exp((it - maxQ) / T) }

        return calculateAndSample(scaled, availableActions)
    }
}