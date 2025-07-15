package io.github.kotlinrl.core.plan

import io.github.kotlinrl.core.*

fun interface TransitionFunction<State, Action> {
    operator fun invoke(state: State, action: Action): Transition<State>
}