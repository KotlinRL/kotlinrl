package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

fun interface TDError<State, Action> {
    operator fun invoke(
        q: QFunction<State, Action>,
        t: Transition<State, Action>,
        aPrime: Action?,
        gamma: Double,
        done: Boolean
    ): Double
}
