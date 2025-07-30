package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

data class EpisodeStats<State, Action>(
    val trajectory: Trajectory<State, Action>,
    val episode: Int,
    val totalReward: Double = trajectory.sumOf { it.reward },
    val steps: Int,
    val reachedGoal: Boolean = false,
    val truncated: Boolean = trajectory.last().truncated,
    val info: Map<String, Any?> = emptyMap()
)