package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

fun interface TDVError<State, Action> {
    operator fun invoke(
        V: ValueFunction<State>,
        t: Transition<State, Action>,
        gamma: Double
    ): Double
}
