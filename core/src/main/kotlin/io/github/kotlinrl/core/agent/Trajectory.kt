package io.github.kotlinrl.core.agent

data class Trajectory<State, Action>(
    val state: State,
    val action: Action,
    val reward: Double,
    val nextState: State,
    val terminated: Boolean,
    val truncated: Boolean,
    val info: Map<String, String>
)