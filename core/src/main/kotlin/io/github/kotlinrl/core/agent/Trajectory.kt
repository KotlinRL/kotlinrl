package io.github.kotlinrl.core.agent

data class Trajectory<State, Action>(
    val nextState: State,
    val reward: Double,
    val terminated: Boolean,
    val truncated: Boolean,
    val info: Map<String, String>,
    val action: Action,
    val state: State
)