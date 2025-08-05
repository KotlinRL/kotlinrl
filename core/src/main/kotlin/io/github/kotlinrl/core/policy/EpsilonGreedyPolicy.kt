package io.github.kotlinrl.core.policy

import kotlin.random.*

class EpsilonGreedyPolicy<State, Action>(
    override val q: EnumerableQFunction<State, Action>,
    override val stateActions: StateActions<State, Action>,
    private val epsilon: ParameterSchedule,
    private val rng: Random = Random.Default
) : QFunctionPolicy<State, Action> {
    private val randomPolicy = RandomPolicy(stateActions, rng)
    private val greedyPolicy = GreedyPolicy(q, stateActions)

    override fun invoke(state: State): Action =
        if (rng.nextDouble() < epsilon()) {
            randomPolicy(state)
        } else {
            greedyPolicy(state)
        }

    override fun improve(q: EnumerableQFunction<State, Action>): Policy<State, Action> =
        EpsilonGreedyPolicy(
            q = q,
            stateActions = stateActions,
            epsilon = epsilon,
            rng = rng
        )
}