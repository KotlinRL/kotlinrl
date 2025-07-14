package io.github.kotlinrl.core.plan

fun interface TransitionFunction<State, Action> {
    operator fun invoke(state: State, action: Action): State
}