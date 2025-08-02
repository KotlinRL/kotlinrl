package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*

fun interface TrajectoryObserver<State, Action> {
    operator fun invoke(trajectory: Trajectory<State, Action>, episode: Int)
}