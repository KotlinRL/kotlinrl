package io.github.kotlinrl.core.train

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

    val totalGoalFailureEpisodes: List<Int> = episodeStats.filter { !it.reachedGoal && !it.truncated }.map { it.episode },
    val totalGoalFailureCount: Int = totalGoalFailureEpisodes.size,
    val totalGoalFailureRate: Double = totalGoalFailureCount.toDouble() / totalEpisodes,

    val allSucceededOrTruncated: Boolean = totalGoalFailureCount == 0
) {
    fun takeLast(n: Int): TrainingResult = TrainingResult(episodeStats.takeLast(n))
}