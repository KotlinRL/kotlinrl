package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.agent.*

data class EpisodeStats<State, Action>(
    val episode: Int,
    val totalReward: Double,
    val steps: Int,
    val experiences: List<Experience<State, Action>>,
)