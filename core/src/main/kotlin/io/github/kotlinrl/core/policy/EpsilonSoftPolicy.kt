package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable
import kotlin.random.*

class EpsilonSoftPolicy<State, Action>(
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val qTable: QTable<State, Action>,
    private val epsilon: ExplorationFactor,
    rng: Random = Random.Default
) : ProbabilisticPolicy<State, Action>(rng) {

    override fun invoke(state: State): Action {
        val availableActions = stateActionListProvider(state)
        require(availableActions.isNotEmpty()) { "No available actions for state: $state" }

        val greedy = availableActions.maxByOrNull { qTable[state, it] }
            ?: error("Unable to determine greedy action.")

        val eps = epsilon()
        val uniformProb = eps / availableActions.size
        val greedyProb = 1.0 - eps + uniformProb

        val probabilities = availableActions.map { action ->
            if (action == greedy) greedyProb else uniformProb
        }

        return calculateAndSample(probabilities, availableActions)
    }
}