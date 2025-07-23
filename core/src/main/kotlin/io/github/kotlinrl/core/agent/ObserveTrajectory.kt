package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*

fun interface ObserveTrajectory<State, Action> : LearningBehavior<State, Action> {
    operator fun invoke(trajectory: Trajectory<State, Action>, episode: Int)
}