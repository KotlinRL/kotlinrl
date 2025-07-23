package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.*

interface Agent<State, Action> {
    val id: String
    fun act(state: State): Action
    fun observe(transition: Transition<State, Action>)
    fun observe(trajectory: Trajectory<State, Action>, episode: Int)
}