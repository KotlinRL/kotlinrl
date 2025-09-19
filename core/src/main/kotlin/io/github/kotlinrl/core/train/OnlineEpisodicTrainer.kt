package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.*

/**
 * Manages the episodic training of a reinforcement learning agent in an environment.
 *
 * This class facilitates the learning process of an agent by executing a series of episodes
 * in a given environment. Each episode consists of interactions between the agent and the
 * environment, where the agent observes the state, selects actions, and receives feedback
 * in the form of rewards. The training process continues until a defined stopping condition
 * is met.
 *
 * Generic Parameters:
 * @param State The type representing the state in the environment.
 * @param Action The type representing the actions performed by the agent.
 *
 * @property env The environment in which the agent interacts during training episodes.
 *               It provides observations, updates through actions, and rewards.
 * @property agent The reinforcement learning agent being trained.
 *                 The agent learns from the state transitions and rewards provided by the environment.
 * @property successfulTermination A functional interface defining the criteria for successful episode termination.
 *                                  Used to evaluate whether a specific state transition meets success conditions.
 * @property closeOnSuccess Whether the environment should be closed automatically upon meeting a success condition.
 *                          Defaults to `false`.
 * @property warnOnTruncationOrMax Whether a warning should be displayed if an episode exceeds the maximum step limit
 *                                 or truncates. Defaults to `false`.
 * @property maxStepsPerEpisode The maximum number of steps allowed in a single episode.
 *                              If this limit is reached, the episode is truncated. Defaults to 10,000.
 * @property callbacks A list of callbacks invoked during the start and end of episodes.
 *                     Provides hooks for monitoring or reacting to training progress.
 */
class OnlineEpisodicTrainer<State, Action>(
    private val env: Env<State, Action, *, *>,
    private val agent: Agent<State, Action>,
    private val successfulTermination: SuccessfulTermination<State, Action>,
    private val closeOnSuccess: Boolean = false,
    private val warnOnTruncationOrMax: Boolean = false,
    private val maxStepsPerEpisode: Int = 10_000,
    private val callbacks: List<EpisodeCallback<State, Action>> = emptyList(),
) : Trainer {

    /**
     * Trains the agent in the given environment using the specified stop condition.
     *
     * This method performs a sequence of training episodes during which the agent interacts
     * with the environment to learn and improve its behavior. The training process continues
     * until the specified stop condition is met. Each episode is terminated when the agent
     * reaches a terminal state, exceeds the maximum allowed steps, or encounters an exception.
     *
     * @param stopCondition A condition that determines when to stop the training process.
     *                       It is a functional interface invoked with the current
     *                       `TrainingResult` at the end of each episode.
     * @return A `TrainingResult` encapsulating the statistics and outcomes of the episodes
     *         executed during the training process.
     */
    override fun train(stopCondition: TrainingStopCondition): TrainingResult {
        val episodeStats = mutableListOf<EpisodeStats<State, Action>>()
        var episode = 1

        while (true) {
            callbacks.forEach { it.onEpisodeStart(episode) }

            val transitions = mutableListOf<Transition<State, Action>>()

            var state = env.reset().state
            var totalReward = 0.0
            var done = false
            var exception: Throwable? = null

            while (!done && transitions.size < maxStepsPerEpisode) {
                try {
                    val action = agent.act(state)
                    val stepResult = env.step(action)
                    totalReward += stepResult.reward

                    val transition = Transition(
                        state = state,
                        action = action,
                        nextState = stepResult.state,
                        reward = stepResult.reward,
                        terminated = stepResult.terminated,
                        truncated = stepResult.truncated,
                        info = stepResult.info
                    )

                    agent.observe(transition)

                    transitions += transition
                    state = transition.nextState
                    done = transition.done
                } catch (e: Exception) {
                    done = true
                    exception = e
                }
            }

            val stats = EpisodeStats(
                trajectory = transitions,
                episode = episode,
                reachedGoal = transitions.lastOrNull()?.let { successfulTermination(it) } ?: false,
                info = exception?.let { mapOf("exception" to it) } ?: emptyMap()
            )
            agent.observe(transitions, episode)
            if(!transitions.last().terminated && warnOnTruncationOrMax) {
                if (transitions.last().truncated) {
                    println("Episode $episode truncated=${transitions.last().truncated}")
                } else {
                    println("Episode $episode failed to terminate and was cut short. maxStepsPerEpisode=$maxStepsPerEpisode")
                }
            }
            episodeStats += stats
            val result = TrainingResult(episodeStats)
            callbacks.forEach { it.onEpisodeEnd(result) }

            if (stopCondition(result)) {
                if (closeOnSuccess) env.close()
                return result
            }
            episode++
        }
    }
}
