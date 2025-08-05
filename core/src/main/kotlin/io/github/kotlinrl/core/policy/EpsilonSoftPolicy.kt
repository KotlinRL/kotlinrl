package io.github.kotlinrl.core.policy

import kotlin.random.*

class EpsilonSoftPolicy<State, Action>(
    override val Q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val epsilon: ParameterSchedule,
    rng: Random
) : StochasticPolicy<State, Action>(rng) {

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActions(state)
        val greedyAction = Q.bestAction(state)
        val n = actions.size
        val epsilon = epsilon()

        return actions.map { action ->
            val prob = if (action == greedyAction) {
                (1 - epsilon) + (epsilon / n)
            } else {
                epsilon / n
            }
            action to prob
        }
    }

    override fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        EpsilonSoftPolicy(
            Q = Q,
            epsilon = epsilon,
            rng = rng,
            stateActions = stateActions
        )
}