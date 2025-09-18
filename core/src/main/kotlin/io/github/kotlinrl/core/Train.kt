package io.github.kotlinrl.core

import kotlin.math.*

typealias OnlineEpisodicTrainer<State, Action> = io.github.kotlinrl.core.train.OnlineEpisodicTrainer<State, Action>
typealias EpisodeCallback<State, Action> = io.github.kotlinrl.core.train.EpisodeCallback<State, Action>
typealias Trainer = io.github.kotlinrl.core.train.Trainer
typealias TrainingResult = io.github.kotlinrl.core.train.TrainingResult
typealias TrainingStopCondition = io.github.kotlinrl.core.train.TrainingStopCondition
typealias SuccessfulTermination<State, Action> = io.github.kotlinrl.core.train.SuccessfulTermination<State, Action>
typealias EpisodeStats<State, Action> = io.github.kotlinrl.core.train.EpisodeStats<State, Action>

/**
 * Creates and configures an episodic trainer for reinforcement learning by connecting an agent with an environment and managing training episodes.
 *
 * @param State The type of state in the environment.
 * @param Action The type of action performed by the agent.
 * @param env The environment within which the agent interacts, learns, and explores.
 * @param agent The reinforcement learning agent responsible for choosing actions and learning from the environment transitions.
 * @param maxStepsPerEpisode The maximum number of steps allowed in a single training episode. Defaults to Int.MAX_VALUE.
 * @param successfulTermination A function to evaluate if the current episode has been successfully terminated. This is a custom condition beyond environment termination signals.
 * @param closeOnSuccess Whether to close the environment automatically after a successful termination condition is met. Defaults to false.
 * @param callbacks A list of episode callbacks that observe and handle training events, such as episode start, step transitions, or completion. Defaults to an empty list.
 * @return A trainer instance that manages the episodic reinforcement learning process.
 */
fun <State, Action> episodicTrainer(
    env: Env<State, Action, *, *>,
    agent: Agent<State, Action>,
    maxStepsPerEpisode: Int = Int.MAX_VALUE,
    successfulTermination: SuccessfulTermination<State, Action>,
    closeOnSuccess: Boolean = false,
    callbacks: List<EpisodeCallback<State, Action>> = emptyList()
): Trainer = OnlineEpisodicTrainer(
    env = env,
    agent = agent,
    maxStepsPerEpisode = maxStepsPerEpisode,
    successfulTermination = successfulTermination,
    closeOnSuccess = closeOnSuccess,
    callbacks = callbacks
)

/**
 * Checks if the average reward of the training exceeds a specified target value
 * after a minimum number of episodes.
 *
 * @param minEpisodes The minimum number of episodes that must be completed
 * before evaluating the average reward.
 * @param target The target average reward that should be exceeded to satisfy the condition.
 * @return A boolean condition that is true if the total average reward exceeds the target
 * after the minimum number of episodes, and false otherwise.
 */
fun averageRewardGreaterThan(minEpisodes: Int, target: Double) = stopCondition {
    if (it.totalEpisodes < minEpisodes) return@stopCondition false
    val condition = it.totalAverageReward > target
    if (condition) println("Average reward at episode ${it.lastEpisode} reached: ${it.totalAverageReward}")
    condition
}

/**
 * Determines if the goal success rate within a sliding window of episodes exceeds a specified target.
 * This function acts as a stopping condition for training when the success rate of reaching the goal,
 * evaluated over the most recent episodes, surpasses a given threshold.
 *
 * @param minEpisodes The minimum number of total episodes required before the condition starts evaluating.
 * @param windowSize The number of most recent episodes to consider for computing the goal success rate.
 * @param target The target success rate threshold that, when exceeded, triggers the condition.
 */
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

/**
 * Creates a stop condition to determine if there has been no significant improvement in rewards over a specified number of episodes.
 *
 * @param minEpisodes The minimum number of episodes required before evaluating the stop condition.
 * @param windowSize The number of recent episodes to consider for calculating average rewards.
 * @param tolerance The threshold for the difference between recent and previous average rewards to consider it as no significant improvement. Defaults to 1e-4.
 */
fun noRecentImprovementAfter(minEpisodes: Int, windowSize: Int, tolerance: Double = 1e-4) = stopCondition {
    if (it.totalEpisodes < minEpisodes) return@stopCondition false
    val rewards = it.totalRewardsList.drop(minEpisodes)
    if (rewards.size < windowSize * 2) return@stopCondition false

    val recentAvg = rewards.takeLast(windowSize).average()
    val previousAvg = rewards.dropLast(windowSize).takeLast(windowSize).average()

    val condition = (recentAvg - previousAvg) < tolerance
    if (condition) println("No significant improvement in ${windowSize} episodes since episode ${it.lastEpisode - windowSize}")
    condition
}

/**
 * Checks if a specified number of consecutive episodes have successfully reached the goal condition.
 *
 * @param threshold The number of consecutive episodes that must reach their goal to meet the stop condition.
 */
fun consecutiveGoalSuccesses(threshold: Int) = stopCondition {
    val recent = it.episodeStats.takeLast(threshold)
    val condition = recent.size == threshold && recent.all { it.reachedGoal }
    if (condition) println("Reached $threshold consecutive goal successes at episode ${it.lastEpisode}")
    condition
}

/**
 * Creates a stop condition that checks if the cumulative maximum reward has reached the specified target value.
 *
 * @param target The target value for the total maximum reward. The condition will stop training when this value is reached or exceeded.
 */
fun maxRewardReached(target: Double) = stopCondition {
    val condition = it.totalMaxReward >= target
    if (condition) println("Max reward at episode ${it.lastEpisode} reached: ${it.totalMaxReward}")
    condition
}

/**
 * Defines a stop condition based on the moving average of the rewards over a specified window size.
 * The condition is satisfied when the moving average of the rewards in the specified window exceeds the given threshold.
 *
 * @param window The size of the window over which the moving average of the rewards is computed.
 * @param threshold The threshold value that the moving average of the rewards must exceed to meet the stop condition.
 */
fun movingAverageRewardGreaterThan(window: Int, threshold: Double) = stopCondition {
    val recent = it.totalRewardsList.takeLast(window)
    val condition = recent.size == window && recent.average() > threshold
    if (condition) println("Average reward at episode ${it.lastEpisode} above threshold: ${recent.average()}")
    condition
}

/**
 * Creates a training stop condition based on the maximum number of episodes.
 * The training will stop once the total number of episodes reaches or exceeds the specified maximum.
 *
 * @param max The maximum number of episodes allowed before stopping the training.
 */
fun maxEpisodes(max: Int) = stopCondition {
    val condition = it.totalEpisodes >= max
    if (condition) println("Max episodes reached: ${it.totalEpisodes}")
    condition
}

/**
 * Creates a stopping condition that evaluates if the variance of the rewards in the most recent window
 * of episodes is below a specified threshold.
 *
 * @param threshold The variance threshold that must not be exceeded for the stopping condition to be met.
 * @param window The number of most recent episodes to consider for calculating the variance. Defaults to 100.
 */
fun rewardVarianceBelow(threshold: Double, window: Int = 100) = stopCondition {
    val recent = it.totalRewardsList.takeLast(window)
    val condition = recent.size == window && recent.variance() < threshold
    if (condition) println("Reward variance at episode ${it.lastEpisode} below threshold: ${recent.variance()}")
    condition
}

/**
 * Defines a stop condition for a training process where the condition is met
 * when the first goal is achieved. The condition checks if the total count
 * of successful goals is greater than zero.
 *
 * If the condition is satisfied, a message indicating that the first goal
 * has been reached is printed, along with the episode where it occurred.
 *
 * @return A training stop condition that evaluates if the first goal has been achieved.
 */
fun firstGoalReached() = stopCondition {
    val condition = it.totalGoalSuccessCount > 0
    if (condition) println("First goal reached at episode ${it.lastEpisode}.")
    condition
}

/**
 * Defines the stopping condition for the training process.
 *
 * @param condition the condition that determines when the training should stop
 * @return the given TrainingStopCondition
 */
fun stopCondition(condition: TrainingStopCondition) = condition
fun <State, Action> printEpisodeStart(
    printEvery: Int
): EpisodeCallback<State, Action> = onEpisodeStart { episode ->
    if (episode % printEvery == 0) println("Starting episode $episode")
}

/**
 * Creates an episode callback that logs the total number of transitions after every specified number of episodes.
 *
 * @param printEvery the interval at which the total transitions are printed. For example, if set to 10,
 * it will log the total transitions every 10 episodes.
 * @return an [EpisodeCallback] that logs episode progress based on the configured interval.
 */
fun <State, Action> printEpisodeTotalTransitions(
    printEvery: Int
): EpisodeCallback<State, Action> = onEpisodeEnd {
    if (it.lastEpisode % printEvery == 0)
        println("Finished episode ${it.lastEpisode}, ${it.lastEpisodeSteps} transitions.")
}

/**
 * Creates an episode callback that prints a message when a goal is reached in an episode.
 * The message is printed only if the goal is reached and the episode number is a multiple of the specified interval.
 *
 * @param printEvery The interval at which to print the goal-reached message. Defaults to 1.
 * @return An episode callback that prints the message when the goal is reached.
 */
fun <State, Action> printEpisodeOnGoalReached(printEvery: Int = 1): EpisodeCallback<State, Action> = onEpisodeEnd {
    if (it.lastEpisodeReachedGoal && it.lastEpisode % printEvery == 0)
        println("Goal reached in episode ${it.lastEpisode}.")
}

/**
 * Constructs an instance of `EpisodeCallback` where the `onEpisodeStart` method
 * invokes the provided callback function.
 *
 * The returned callback can be used to execute custom logic at the start of a training
 * episode, such as logging, initialization, or monitoring.
 *
 * @param f A function that is invoked at the start of each episode. It accepts the
 *          episode number as an argument.
 * @return An `EpisodeCallback` implementation that triggers the provided function
 *         when the `onEpisodeStart` callback is invoked during training.
 */
fun <State, Action> onEpisodeStart(
    f: (episode: Int) -> Unit
): EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeStart(episode: Int) = f(episode)
}

/**
 * Creates an `EpisodeCallback` that executes a specified block of code at the end of each episode.
 *
 * This function can be used to handle or monitor the result of an episode when it ends during
 * the training process. The provided `block` is executed with the episode's training result as input.
 *
 * @param block A lambda function to be executed at the end of each episode. It receives a `TrainingResult`
 *              which contains detailed metrics and outcomes of the episode.
 * @return An `EpisodeCallback` instance that triggers the specified `block` when the `onEpisodeEnd`
 *         callback is invoked.
 */
fun <State, Action> onEpisodeEnd(
    block: (result: TrainingResult) -> Unit
): EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeEnd(result: TrainingResult) = block(result)
}

/**
 * Calculates the variance of the elements in the list. Variance is a measure of
 * the dispersion of the elements around their mean, defined as the average of
 * the squared differences from the mean.
 *
 * @return The variance of the list elements as a Double value.
 */
fun List<Double>.variance(): Double {
    val mean = average()
    return map { (it - mean).pow(2) }.average()
}


/**
 * Combines the current `TrainingStopCondition` with another using a logical AND operation.
 *
 * The resulting `TrainingStopCondition` will evaluate to `true` if both the current condition
 * and the provided `other` condition evaluate to `true` for a given `TrainingResult`.
 * This allows for the creation of composite stopping criteria that require multiple
 * independent conditions to be satisfied simultaneously.
 *
 * @param other Another `TrainingStopCondition` to combine with the current condition.
 * @return A new `TrainingStopCondition` that represents the logical AND of the current and provided conditions.
 */
fun TrainingStopCondition.and(other: TrainingStopCondition) = TrainingStopCondition { this(it) && other(it) }
/**
 * Combines this `TrainingStopCondition` with another `TrainingStopCondition` using a logical OR operation.
 *
 * The resulting `TrainingStopCondition` will return `true` if either this condition or the provided condition
 * evaluates to `true` for a given `TrainingResult`. This allows the composition of multiple stop conditions
 * to dynamically control training termination based on different criteria.
 *
 * @param other The other `TrainingStopCondition` to combine with this condition.
 * @return A new `TrainingStopCondition` that evaluates to `true` if either this condition or the provided
 * condition evaluates to `true` for the same `TrainingResult`.
 */
fun TrainingStopCondition.or(other: TrainingStopCondition) = TrainingStopCondition { this(it) || other(it) }