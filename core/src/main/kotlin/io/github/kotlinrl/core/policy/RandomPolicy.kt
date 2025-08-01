package io.github.kotlinrl.core.policy

import kotlin.random.*

class RandomPolicy<State, Action>(
    private val actionListProvider: StateActionListProvider<State, Action>,
    private val rng: Random = Random.Default
) : Policy<State, Action>, PolicyImprovementStrategy<State, Action> {

    override operator fun invoke(state: State): Action {
        val availableActions = actionListProvider(state)
        return availableActions[rng.nextInt(availableActions.size)]
    }

    override operator fun invoke(q: QFunction<State, Action>): Policy<State, Action> = this
}
