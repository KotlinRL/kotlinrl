package io.github.kotlinrl.core.policy

import kotlin.random.*

class EpsilonSoftPolicy<State, Action>(
    override val q: QFunction<State, Action>,
    private val epsilon: ParameterSchedule,
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random
) : StochasticPolicy<State, Action>(stateActionListProvider, rng), PolicyImprovementStrategy<State, Action>,
    QFunctionPolicy<State, Action> {

    override fun actionScores(state: State): List<Pair<Action, Double>> {
        val actions = stateActionListProvider(state)
        val greedyAction = q.bestAction(state)
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

    override operator fun invoke(q: QFunction<State, Action>): Policy<State, Action> =
        EpsilonSoftPolicy(
            q = q,
            epsilon = epsilon,
            stateActionListProvider = stateActionListProvider,
            rng = rng
        )
}