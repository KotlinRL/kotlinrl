package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

fun interface NStepTDVError<State> {
    operator fun invoke(
        V: ValueFunction<State>,
        t: Trajectory<State, *>,
        gamma: Double
    ): Double
}
