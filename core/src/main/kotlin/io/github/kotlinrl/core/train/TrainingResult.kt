package io.github.kotlinrl.core.train

data class TrainingResult(
    val episodeStats: List<EpisodeStats<*, *>>,

    val lastEpisodeStats: EpisodeStats<*, *> = episodeStats.last(),
    val lastEpisode: Int = lastEpisodeStats.episode,
    val lastEpisodeTotalReward: Double = lastEpisodeStats.totalReward,
    val lastEpisodeReachedGoal: Boolean = lastEpisodeStats.reachedGoal,
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

    val goalSuccessEpisodes: List<Int> = episodeStats.filter { it.reachedGoal }.map { it.episode },
    val goalSuccessCount: Int = goalSuccessEpisodes.size,
    val goalSuccessRate: Double = goalSuccessCount.toDouble() / totalEpisodes,
    val firstSuccessEpisode: Int? = goalSuccessEpisodes.firstOrNull(),

    val truncatedEpisodes: List<Int> = episodeStats.filter { it.truncated }.map { it.episode },
    val truncatedEpisodeCount: Int = episodeStats.filter { it.truncated }.size,
    val truncatedEpisodeRate: Double = truncatedEpisodeCount.toDouble() / totalEpisodes,
)