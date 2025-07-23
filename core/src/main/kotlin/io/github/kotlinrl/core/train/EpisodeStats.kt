package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

data class EpisodeStats<State, Action>(
    val episode: Int,
    val totalReward: Double,
    val steps: Int,
    val trajectory: Trajectory<State, Action>
)