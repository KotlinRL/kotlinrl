package io.github.kotlinrl.core.policy

import kotlin.random.*

class RandomPolicy<State, Action>(
    private val stateActions: StateActions<State, Action>,
    private val rng: Random = Random.Default
) : Policy<State, Action> {

    override operator fun invoke(state: State): Action {
        val actions = stateActions(state)
        return actions[rng.nextInt(actions.size)]
    }
}
