package io.github.kotlinrl.core.policy

import kotlin.random.*

class EpsilonGreedyPolicy<State, Action>(
    override val q: QFunction<State, Action>,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    private val epsilon: ParameterSchedule,
    private val rng: Random = Random.Default
) : QFunctionPolicy<State, Action>, PolicyImprovementStrategy<State, Action> {
    private val randomPolicy = RandomPolicy(stateActionListProvider, rng)
    private val greedyPolicy = GreedyPolicy(q)

    override fun invoke(state: State): Action =
        if (rng.nextDouble() < epsilon()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }

    override operator fun invoke(q: QFunction<State, Action>): Policy<State, Action> =
        EpsilonGreedyPolicy(
            q = q,
            stateActionListProvider = stateActionListProvider,
            epsilon = epsilon,
            rng = rng
        )
}