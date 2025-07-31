package io.github.kotlinrl.core

import kotlin.math.pow

typealias EpisodeTrainer<State, Action> = io.github.kotlinrl.core.train.EpisodeTrainer<State, Action>
typealias EpisodeCallback<State, Action> = io.github.kotlinrl.core.train.EpisodeCallback<State, Action>
typealias Trainer = io.github.kotlinrl.core.train.Trainer
typealias Trajectory<State, Action> = List<Transition<State, Action>>
typealias TrainingResult = io.github.kotlinrl.core.train.TrainingResult
typealias TrainingStopCondition = io.github.kotlinrl.core.train.TrainingStopCondition
typealias SuccessfulTermination<State, Action> = io.github.kotlinrl.core.train.SuccessfulTermination<State, Action>
typealias Env<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.env.Env<State, Action, ObservationSpace, ActionSpace>
typealias EpisodeStats<State, Action> = io.github.kotlinrl.core.train.EpisodeStats<State, Action>

fun <State, Action> episodicTrainer(
    env: Env<State, Action, *, *>,
    agent: Agent<State, Action>,
    maxStepsPerEpisode: Int = 10_000,
    successfulTermination: SuccessfulTermination<State, Action>,
    closeOnSuccess: Boolean = false,
    callbacks: List<EpisodeCallback<State, Action>> = emptyList()
): Trainer = EpisodeTrainer(
    env = env,
    agent = agent,
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = successfulTermination,
    closeOnSuccess = closeOnSuccess,
    callbacks = callbacks
)

fun averageRewardGreaterThan(minEpisodes: Int, target: Double) = stopCondition {
    if (it.totalEpisodes < minEpisodes) return@stopCondition false
    val condition = it.totalAverageReward > target
    if(condition) println("Average reward at episode ${it.lastEpisode} reached: ${it.totalAverageReward}")
    condition
}

fun goalSuccessRateGreaterThanAfter(
    minEpisodes: Int,
    windowSize: Int,
    target: Double
) = stopCondition {
    if (it.totalEpisodes < minEpisodes + windowSize) return@stopCondition false

    val relevant = it.episodeStats.drop(it.totalEpisodes - windowSize)
    val rate = relevant.count { it.reachedGoal }.toDouble() / windowSize

    val condition = rate > target
    if (condition) println("Goal success rate in last $windowSize episodes at episode ${it.lastEpisode} reached: $rate")
    condition
}

fun noRecentImprovementAfter(minEpisodes: Int, windowSize: Int, tolerance: Double = 1e-4) = stopCondition {
    if (it.totalEpisodes < minEpisodes) return@stopCondition false
    val rewards = it.totalRewardsList.drop(minEpisodes)
    if (rewards.size < windowSize * 2) return@stopCondition false

    val recentAvg = rewards.takeLast(windowSize).average()
    val previousAvg = rewards.dropLast(windowSize).takeLast(windowSize).average()

    val condition = (recentAvg - previousAvg) < tolerance
    if(condition) println("No significant improvement in ${windowSize} episodes since episode ${it.lastEpisode - windowSize}")
    condition
}

fun consecutiveGoalSuccesses(threshold: Int) = stopCondition {
    val recent = it.episodeStats.takeLast(threshold)
    val condition = recent.size == threshold && recent.all { it.reachedGoal }
    if(condition) println("Reached $threshold consecutive goal successes at episode ${it.lastEpisode}")
    condition
}

fun maxRewardReached(target: Double) = stopCondition {
    val condition = it.totalMaxReward >= target
    if(condition) println("Max reward at episode ${it.lastEpisode} reached: ${it.totalMaxReward}")
    condition
}

fun movingAverageRewardGreaterThan(window: Int, threshold: Double) = stopCondition {
    val recent = it.totalRewardsList.takeLast(window)
    val condition = recent.size == window && recent.average() > threshold
    if(condition) println("Average reward at episode ${it.lastEpisode} above threshold: ${recent.average()}")
    condition
}

fun maxEpisodes(max: Int) = stopCondition {
    val condition = it.totalEpisodes >= max
    if(condition) println("Max episodes reached: ${it.totalEpisodes}")
    condition
}

fun rewardVarianceBelow(threshold: Double, window: Int = 100) = stopCondition {
    val recent = it.totalRewardsList.takeLast(window)
    val condition = recent.size == window && recent.variance() < threshold
    if(condition) println("Reward variance at episode ${it.lastEpisode} below threshold: ${recent.variance()}")
    condition
}

fun firstGoalReached() = stopCondition {
    val condition = it.totalGoalSuccessCount > 0
    if(condition) println("First goal reached at episode ${it.lastEpisode}.")
    condition
}

fun stopCondition(condition: TrainingStopCondition) = condition

fun List<Double>.variance(): Double {
    val mean = average()
    return map { (it - mean).pow(2) }.average()
}


fun TrainingStopCondition.and(other: TrainingStopCondition) = TrainingStopCondition { this(it) && other(it) }
fun TrainingStopCondition.or(other: TrainingStopCondition) = TrainingStopCondition { this(it) || other(it) }