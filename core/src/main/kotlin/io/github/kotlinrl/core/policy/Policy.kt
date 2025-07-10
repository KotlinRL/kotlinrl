package io.github.kotlinrl.core.policy

fun interface Policy<State, Action>{
    operator fun invoke(state: State): Action
}