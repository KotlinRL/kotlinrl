package io.github.kotlinrl.core.policy

interface QFunctionPolicy<State, Action> : Policy<State, Action> {
    val q: QFunction<State, Action>
}