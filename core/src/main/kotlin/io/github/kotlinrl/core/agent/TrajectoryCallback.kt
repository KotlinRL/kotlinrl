package io.github.kotlinrl.core.agent

interface TrajectoryCallback<State, Action> {
    fun before() = { }
    fun after(trajectory: Trajectory<State, Action>) = { }
}