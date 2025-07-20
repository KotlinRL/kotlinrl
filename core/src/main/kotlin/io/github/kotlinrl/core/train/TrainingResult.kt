package io.github.kotlinrl.core.train

data class TrainingResult(
    val episodeRewards: List<Double>,
    val totalEpisodes: Int = episodeRewards.size,
    val averageReward: Double = episodeRewards.average()
)