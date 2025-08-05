package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.SoftmaxPolicy
import kotlin.math.*
import kotlin.random.*

class SoftmaxPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val temperature: ParameterSchedule,
    rng: Random
) : StochasticPolicy<State, Action>(rng) {

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val temperature = temperature()
        val actions = stateActions(state)
        return actions.map { action ->
            action to exp(Q[state, action] / temperature)
        }
    }

    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        SoftmaxPolicy(Q, stateActions, temperature, rng)
}