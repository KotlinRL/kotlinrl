package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.env.*

data class EpisodeStats<State, Action>(
    val episode: Int,
    val totalReward: Double,
    val steps: Int,
    val transitions: List<Transition<State>>,
    val actions: List<Action>
)