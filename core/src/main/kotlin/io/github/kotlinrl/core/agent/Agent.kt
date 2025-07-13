package io.github.kotlinrl.core.agent

interface Agent<State, Action> {
    val id: String
    fun act(state: State): Action
    fun observe(trajectory: Trajectory<State, Action>)
}