package io.github.kotlinrl.core.policy

fun interface QFunction<State, Action> {
    operator fun invoke(state: State, action: Action): Double
}