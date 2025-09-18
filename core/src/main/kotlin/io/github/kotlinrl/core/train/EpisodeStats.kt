package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

/**
 * Represents statistical data and outcome information for a single episode during training.
 *
 * This data class encapsulates a variety of metrics and flags related to the performance
 * and progression of an episode. It is commonly used to record and analyze information
 * about the agent's behavior, environment interactions, and training process results.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the action taken by the agent.
 * @property trajectory A collection of transitions (state-action-nextState-reward) that occurred during the episode.
 * @property episode The episode number, identifying which episode this information corresponds to.
 * @property totalReward The cumulative reward collected by the agent throughout the episode.
 * @property steps The number of steps taken within the episode.
 * @property reachedGoal Indicates whether the agent successfully achieved the goal during the episode.
 * @property truncated Indicates whether the episode was truncated before reaching a terminal state.
 * @property info A map of additional information about the episode, which can contain custom metadata or debugging details.
 */
data class EpisodeStats<State, Action>(
    val trajectory: Trajectory<State, Action>,
    val episode: Int,
    val totalReward: Double = trajectory.sumOf { it.reward },
    val steps: Int = trajectory.size,
    val reachedGoal: Boolean = false,
    val truncated: Boolean = trajectory.lastOrNull()?.truncated ?: false,
    val info: Map<String, Any?> = emptyMap()
)