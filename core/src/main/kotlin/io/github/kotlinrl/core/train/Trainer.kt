package io.github.kotlinrl.core.train

/**
 * Represents a trainer interface responsible for orchestrating the training process for a
 * reinforcement learning agent. Implementations of this interface contain the logic necessary
 * for training an agent, interacting with an external environment, and collecting training statistics.
 */
interface Trainer {
    /**
     * Executes the training process for a reinforcement learning agent until the specified stop condition is met.
     *
     * This method manages the sequential execution of training episodes, during which the agent interacts with
     * an environment to learn and improve its performance. At the end of each episode, the training status is evaluated
     * against the provided stop condition to determine whether to continue or terminate the training process.
     *
     * @param stopCondition A functional interface invoked with the current `TrainingResult` at the end of each episode.
     *                       The training process will stop when this condition evaluates to `true`.
     * @return A `TrainingResult` object containing statistics and outcomes of the training process, including details
     *         about the episodes executed and cumulative metrics.
     */
    fun train(stopCondition: TrainingStopCondition): TrainingResult
}