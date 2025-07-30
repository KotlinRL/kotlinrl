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

fun averageRewardGreaterThan(minEpisodes: Int, target: Double) = TrainingStopCondition {
    if (it.totalEpisodes < minEpisodes) return@TrainingStopCondition false
    val condition = it.averageReward > target
    if(condition) println("Average reward at episode ${it.episodeStats.last().episode} reached: ${it.averageReward}")
    condition
}

fun goalSuccessRateGreaterThanAfter(
    minEpisodes: Int,
    windowSize: Int,
    target: Double
) = TrainingStopCondition {
    if (it.totalEpisodes < minEpisodes + windowSize) return@TrainingStopCondition false

    val relevant = it.episodeStats.drop(it.totalEpisodes - windowSize)
    val rate = relevant.count { it.reachedGoal }.toDouble() / windowSize

    val condition = rate > target
    if (condition) println("Goal success rate in last $windowSize episodes at episode ${it.episodeStats.last().episode} reached: $rate")
    condition
}

fun noRecentImprovementAfter(minEpisodes: Int, windowSize: Int, tolerance: Double = 1e-4) = TrainingStopCondition {
    if (it.totalEpisodes < minEpisodes) return@TrainingStopCondition false
    val rewards = it.episodeRewards.drop(minEpisodes)
    if (rewards.size < windowSize * 2) return@TrainingStopCondition false

    val recentAvg = rewards.takeLast(windowSize).average()
    val previousAvg = rewards.dropLast(windowSize).takeLast(windowSize).average()

    val condition = (recentAvg - previousAvg) < tolerance
    if(condition) println("No significant improvement in ${windowSize} episodes since episode ${it.episodeStats.last().episode - windowSize}")
    condition
}

fun consecutiveGoalSuccesses(threshold: Int) = TrainingStopCondition {
    val recent = it.episodeStats.takeLast(threshold)
    val condition = recent.size == threshold && recent.all { it.reachedGoal }
    if(condition) println("Reached $threshold consecutive goal successes at episode ${it.episodeStats.last().episode}")
    condition
}

fun maxRewardReached(target: Double) = TrainingStopCondition {
    val condition = it.maxReward >= target
    if(condition) println("Max reward at episode ${it.episodeStats.last().episode} reached: ${it.maxReward}")
    condition
}

fun movingAverageRewardGreaterThan(window: Int, threshold: Double) = TrainingStopCondition {
    val recent = it.episodeRewards.takeLast(window)
    val condition = recent.size == window && recent.average() > threshold
    if(condition) println("Average reward at episode ${it.episodeStats.last().episode} above threshold: ${recent.average()}")
    condition
}

fun maxEpisodes(max: Int) = TrainingStopCondition {
    val condition = it.totalEpisodes >= max
    if(condition) println("Max episodes reached: ${it.totalEpisodes}")
    condition
}

fun rewardVarianceBelow(threshold: Double, window: Int = 100) = TrainingStopCondition {
    val recent = it.episodeRewards.takeLast(window)
    val condition = recent.size == window && recent.variance() < threshold
    if(condition) println("Reward variance at episode ${it.episodeStats.last().episode} below threshold: ${recent.variance()}")
    condition
}

fun firstGoalReached() = TrainingStopCondition {
    val condition = it.goalSuccessCount > 0
    if(condition) println("First goal reached at episode ${it.episodeStats.last().episode}.")
    condition
}

fun List<Double>.variance(): Double {
    val mean = average()
    return map { (it - mean).pow(2) }.average()
}


fun TrainingStopCondition.and(other: TrainingStopCondition) = TrainingStopCondition { this(it) && other(it) }
fun TrainingStopCondition.or(other: TrainingStopCondition) = TrainingStopCondition { this(it) || other(it) }