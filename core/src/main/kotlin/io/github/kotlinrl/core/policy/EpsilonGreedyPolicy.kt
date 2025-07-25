package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.*
import kotlin.random.*

class EpsilonGreedyPolicy<State, Action>(
    stateActionListProvider: StateActionListProvider<State, Action>,
    override val qTable: QFunction<State, Action>,
    private val epsilon: ParameterSchedule,
    private val rng: Random = Random.Default
) : QFunctionPolicy<State, Action> {
    private val randomPolicy = randomPolicy(stateActionListProvider, rng)
    private val greedyPolicy = greedyPolicy(qTable)

    override fun invoke(state: State): Action {
        return if (rng.nextDouble() < epsilon()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }
    }
}