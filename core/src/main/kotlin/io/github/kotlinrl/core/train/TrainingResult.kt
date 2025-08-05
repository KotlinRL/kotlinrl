package io.github.kotlinrl.core.train

/**
 * Represents the result of a training process, encapsulating statistical data and metrics
 * gathered across multiple episodes of training.
 *
 * This data class provides aggregated and detailed information about the training process,
 * including statistics for the last episode, cumulative metrics, success and failure rates,
 * and additional metrics that facilitate analysis and evaluation of the training performance.
 *
 * @property episodeStats A list of statistical information for each episode conducted during training.
 * @property lastEpisodeStats Statistics for the last executed episode.
 * @property lastEpisode The number of the last episode.
 * @property lastEpisodeTotalReward The total reward collected during the last episode.
 * @property lastEpisodeReachedGoal Indicates whether the goal was reached in the last episode.
 * @property lastEpisodeTruncated Indicates whether the last episode was truncated before completing.
 * @property lastEpisodeSteps The number of steps taken in the last episode.
 * @property totalEpisodes The total number of episodes conducted during the training.
 * @property totalRewardsList A list of total rewards collected for each episode.
 * @property totalRewards The sum of total rewards collected across all episodes.
 * @property totalAverageReward The average reward collected across all episodes.
 * @property totalMaxReward The maximum reward observed in any single episode.
 * @property totalMinReward The minimum reward observed in any single episode.
 * @property totalStepsList A list of the step counts for each episode.
 * @property totalSteps The total number of steps taken across all episodes.
 * @property totalStepsAverage The average number of steps taken per episode.
 * @property totalMaxSteps The maximum number of steps observed in a single episode.
 * @property totalMinSteps The minimum number of steps observed in a single episode.
 * @property totalGoalSuccessEpisodes A list of episodes where the goal was successfully reached.
 * @property totalGoalSuccessCount The total number of episodes where the goal was reached.
 * @property totalGoalSuccessRate The rate of goal success across all episodes.
 * @property firstSuccessEpisode The first episode where the goal was successfully reached, if any.
 * @property totalTruncatedEpisodes A list of episodes that were truncated without reaching the goal.
 * @property totalTruncatedEpisodeCount The total number of truncated episodes.
 * @property totalTruncatedEpisodeRate The rate of episodes that were truncated.
 * @property totalGoalFailureEpisodes A list of episodes that neither reached the goal nor were truncated.
 * @property totalGoalFailureCount The total number of episodes where the goal was not reached and the episode was not truncated.
 * @property totalGoalFailureRate The rate of goal failure episodes.
 * @property allSucceededOrTruncated Indicates if all episodes either reached the goal or were truncated, with no failures.
 */
data class TrainingResult(
    val episodeStats: List<EpisodeStats<*, *>>,

    val lastEpisodeStats: EpisodeStats<*, *> = episodeStats.last(),
    val lastEpisode: Int = lastEpisodeStats.episode,
    val lastEpisodeTotalReward: Double = lastEpisodeStats.totalReward,
    val lastEpisodeReachedGoal: Boolean = lastEpisodeStats.reachedGoal,
    val lastEpisodeTruncated: Boolean = lastEpisodeStats.truncated,
    val lastEpisodeSteps: Int = lastEpisodeStats.steps,

    val totalEpisodes: Int = episodeStats.size,
    val totalRewardsList: List<Double> = episodeStats.map { it.totalReward },
    val totalRewards: Double = episodeStats.sumOf { it.totalReward },
    val totalAverageReward: Double = totalRewardsList.average(),
    val totalMaxReward: Double = totalRewardsList.maxOrNull() ?: 0.0,
    val totalMinReward: Double = totalRewardsList.minOrNull() ?: 0.0,
    val totalStepsList: List<Int> = episodeStats.map { it.steps },
    val totalSteps: Int = totalStepsList.sum(),
    val totalStepsAverage: Double = totalStepsList.average(),
    val totalMaxSteps: Int = totalStepsList.maxOrNull() ?: 0,
    val totalMinSteps: Int = totalStepsList.minOrNull() ?: 0,

    val totalGoalSuccessEpisodes: List<Int> = episodeStats.filter { it.reachedGoal }.map { it.episode },
    val totalGoalSuccessCount: Int = totalGoalSuccessEpisodes.size,
    val totalGoalSuccessRate: Double = totalGoalSuccessCount.toDouble() / totalEpisodes,
    val firstSuccessEpisode: Int? = totalGoalSuccessEpisodes.firstOrNull(),

    val totalTruncatedEpisodes: List<Int> = episodeStats.filter { !it.reachedGoal && it.truncated }.map { it.episode },
    val totalTruncatedEpisodeCount: Int = totalTruncatedEpisodes.size,
    val totalTruncatedEpisodeRate: Double = totalTruncatedEpisodeCount.toDouble() / totalEpisodes,

    val totalGoalFailureEpisodes: List<Int> = episodeStats.filter { !it.reachedGoal && !it.truncated }
        .map { it.episode },
    val totalGoalFailureCount: Int = totalGoalFailureEpisodes.size,
    val totalGoalFailureRate: Double = totalGoalFailureCount.toDouble() / totalEpisodes,

    val allSucceededOrTruncated: Boolean = totalGoalFailureCount == 0
) {
    fun takeLast(n: Int): TrainingResult = TrainingResult(episodeStats.takeLast(n))
}