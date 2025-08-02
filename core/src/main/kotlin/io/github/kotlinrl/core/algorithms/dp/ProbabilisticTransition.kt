package io.github.kotlinrl.core.algorithms.dp

data class ProbabilisticTransition<State, Action>(
    val state: State,
    val action: Action,
    val reward: Double,
    val nextState: State,
    val probability: Double,
    val done: Boolean
)
