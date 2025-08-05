package io.github.kotlinrl.core.train

/**
 * Interface defining callbacks for monitoring and handling events during training episodes.
 *
 * Implementations of this interface can be used to observe the start and end of individual
 * episodes during the training process. This allows for custom logic, such as logging,
 * metrics collection, or debugging, to be executed at specific stages of the training.
 *
 * Generic parameters:
 * @param State The type representing the state in the environment.
 * @param Action The type representing the action taken by the agent.
 */
interface EpisodeCallback<State, Action> {
    /**
     * Callback method invoked at the start of an episode during training.
     *
     * This method can be used to set up any necessary preconditions, logging mechanisms,
     * or custom logic to execute before the episode begins execution.
     *
     * @param episode The number of the episode that is about to start.
     */
    fun onEpisodeStart(episode: Int) {}

    /**
     * Callback method invoked at the end of an episode during training.
     *
     * This method provides detailed information about the current state of the training
     * process at the end of the episode. It is typically used to analyze or log episode outcomes,
     * such as rewards, steps taken, or whether the goal was achieved.
     *
     * @param result The aggregated results and statistics of the training process up to
     *               and including the end of the current episode. Contains metrics such as
     *               total rewards, steps, success rates, and information about the most
     *               recent episode.
     */
    fun onEpisodeEnd(result: TrainingResult) {}
}