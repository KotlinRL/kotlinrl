package io.github.kotlinrl.core.policy

fun interface Policy<State, Action> {
    operator fun invoke(state: State): Action

    fun improve(Q: EnumerableQFunction<State, Action>): Policy<State, Action> = this
}