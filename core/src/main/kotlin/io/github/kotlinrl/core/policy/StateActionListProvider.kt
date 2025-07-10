package io.github.kotlinrl.core.policy

fun interface StateActionListProvider<State, Action> {
    operator fun invoke(state: State): List<Action>
}