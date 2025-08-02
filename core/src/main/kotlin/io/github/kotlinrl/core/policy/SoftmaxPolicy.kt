package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.SoftmaxPolicy
import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy<State, Action>(
    override val q: QFunction<State, Action>,
    private val temperature: ParameterSchedule,
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random
) : StochasticPolicy<State, Action>(stateActionListProvider, rng), PolicyImprovementStrategy<State, Action>, QFunctionPolicy<State, Action> {

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val temperature = temperature()
        val actions = stateActionListProvider(state)
        return actions.map { action ->
            action to exp(q[state, action] / temperature)
        }
    }

    override operator fun invoke(q: QFunction<State, Action>): Policy<State, Action> =
        SoftmaxPolicy(q, temperature, stateActionListProvider, rng)
}