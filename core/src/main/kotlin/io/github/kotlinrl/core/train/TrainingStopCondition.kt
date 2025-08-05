package io.github.kotlinrl.core.train

/**
 * A functional interface defining the condition for stopping the training process.
 *
 * The `TrainingStopCondition` interface provides a mechanism to evaluate whether
 * a training session should terminate based on the current state of the training.
 * The decision is made by invoking the functional interface with the `TrainingResult`
 * produced after completing an episode.
 *
 * Implementations are expected to provide the logic necessary to determine whether
 * training should stop. For example, it could evaluate specific metrics or thresholds
 * from the `TrainingResult`, such as success rates, episode counts, or average rewards.
 *
 * This interface can be used with training processes, for instance, in implementations
 * of the `Trainer` interface, to dynamically control the length of training sessions.
 *
 * @see TrainingResult
 * @see io.github.kotlinrl.core.train.Trainer
 */
fun interface TrainingStopCondition {
    /**
     * Evaluates the given training result to determine whether the training process should stop.
     *
     * This function is typically invoked at the end of each training episode, allowing for
     * dynamic control over the continuation or termination of the training based on the provided
     * `TrainingResult`. The implementation defines the specific criteria for stopping the training,
     * such as achieving a target success rate or surpassing predefined thresholds.
     *
     * @param result The `TrainingResult` object containing the aggregated data and metrics of
     *               the training process, including details of the latest episode.
     * @return A boolean value indicating whether the training should stop. Returns `true` if
     *         the stopping condition is met; otherwise, returns `false`.
     */
    operator fun invoke(result: TrainingResult): Boolean
}