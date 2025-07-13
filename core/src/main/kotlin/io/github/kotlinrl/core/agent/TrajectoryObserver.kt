package io.github.kotlinrl.core.agent

fun interface TrajectoryObserver<State, Action> {
    operator fun invoke(trajectory: Trajectory<State, Action>)
}