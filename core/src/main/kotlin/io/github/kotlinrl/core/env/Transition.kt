package io.github.kotlinrl.core.env

data class Transition<State>(
    val state: State,
    val reward: Double,
    val terminated: Boolean,
    val truncated: Boolean,
    val info: Map<String, String>
)