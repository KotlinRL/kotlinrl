package io.github.kotlinrl.core.policy

fun interface StateActions<State, Action> {
    operator fun invoke(state: State): List<Action>
}