package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

fun interface NStepTDQError<State, Action> {
    operator fun invoke(
        Q: QFunction<State, Action>,
        t: Trajectory<State, Action>,
        policy: QFunctionPolicy<State, Action>?,
        tailAction: Action?,
        gamma: Double,
    ): Double
}
