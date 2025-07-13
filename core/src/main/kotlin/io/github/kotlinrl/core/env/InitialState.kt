package io.github.kotlinrl.core.env

data class InitialState<State>(
    val state: State,
    val info: Map<String, Any> = mapOf()
)
