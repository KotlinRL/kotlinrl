package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*

/**
 * A wrapper class for an environment that records cumulative statistics for each episode,
 * including the total reward and the episode length, and adds this information to the `info`
 * map returned in the final step of the episode.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the observation space for the state.
 * @param ActionSpace The type of space specifying the allowable actions in the environment.
 * @param env The wrapped environment instance for which episode statistics are recorded.
 */
class RecordEpisodeStatistics<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var episodeReward = 0.0
    private var episodeLength = 0

    /**
     * Resets the environment to its initial state while resetting the recorded episode statistics.
     *
     * This method initializes the environment for a new episode by re-setting the cumulative
     * statistics, such as episode reward and episode length, to zero. It then invokes the
     * wrapped environment's `reset` method to return the initial state of the environment.
     *
     * @param seed An optional random seed for ensuring deterministic environment behavior. If `null`, the default seed or random generator of the environment is used.
     * @param options An optional map of additional parameters to configure the environment reset process. The specific keys and values depend on the wrapped environment implementation
     * .
     * @return The environment's initial state as an `InitialState` object, containing the state and any associated metadata in the `info` map.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        episodeReward = 0.0
        episodeLength = 0
        return env.reset(seed, options)
    }

    /**
     * Executes a single step in the environment while recording episode-level statistics.
     *
     * This method applies the provided action to the environment, updates internal statistics
     * for the current episode (such as total reward and length), and adjusts the `info` object
     * to include episode-level statistics if the episode ends. The statistics reset automatically
     * for new episodes.
     *
     * @param action The action to be performed in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and additional information after the action is applied. This includes
     * episode-level statistics if the episode ends.
     */
    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        episodeReward += t.reward
        episodeLength += 1

        val done = t.terminated || t.truncated
        val newInfo = t.info.toMutableMap()
        if (done) {
            // Gymnasium convention: "episode" is a dict with "r" (reward), "l" (length)
            val episodeStats = mapOf(
                "r" to episodeReward.toString(),
                "l" to episodeLength.toString()
            )
            newInfo["episode"] = episodeStats.toString() // as String, or use JSON if desired

            // Reset counters for next episode
            episodeReward = 0.0
            episodeLength = 0
        }
        return t.copy(info = newInfo)
    }
}
