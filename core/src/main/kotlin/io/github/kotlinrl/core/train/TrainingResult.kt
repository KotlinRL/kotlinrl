package io.github.kotlinrl.core.train

data class TrainingResult(
    val episodeStats: List<EpisodeStats<*, *>>,

    val episodeRewards: List<Double> = episodeStats.map { it.totalReward },
    val totalEpisodes: Int = episodeStats.size,
    val averageReward: Double = episodeRewards.average(),
    val maxReward: Double = episodeRewards.maxOrNull() ?: 0.0,
    val minReward: Double = episodeRewards.minOrNull() ?: 0.0,

    val episodeLengths: List<Int> = episodeStats.map { it.steps },
    val averageEpisodeLength: Double = episodeLengths.average(),
    val maxEpisodeLength: Int = episodeLengths.maxOrNull() ?: 0,
    val minEpisodeLength: Int = episodeLengths.minOrNull() ?: 0,

    val reachedGoalEpisodes: List<Int> = episodeStats.filter { it.reachedGoal }.map { it.episode },
    val goalSuccessCount: Int = reachedGoalEpisodes.size,
    val goalSuccessRate: Double = goalSuccessCount.toDouble() / totalEpisodes,
    val firstSuccessEpisode: Int? = reachedGoalEpisodes.firstOrNull(),

    val truncatedEpisodes: List<Int> = episodeStats.filter { it.truncated }.map { it.episode },
    val truncatedEpisodeCount: Int = episodeStats.filter { it.truncated }.size,
    val truncatedEpisodeRate: Double = truncatedEpisodeCount.toDouble() / totalEpisodes,
)