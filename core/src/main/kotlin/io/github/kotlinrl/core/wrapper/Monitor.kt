package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.io.*

/**
 * A wrapper for an environment that monitors and logs episode statistics, such as rewards and lengths,
 * to a CSV file. This wrapper is useful for tracking performance metrics during agent training or evaluation.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the observation space for the state.
 * @param ActionSpace The type of space specifying the allowable actions in the environment.
 * @param env The underlying environment to be monitored and wrapped.
 * @param logPath The file path where episode statistics will be logged. Defaults to "monitor.csv".
 */
class Monitor<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    logPath: String = "monitor.csv"
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var episodeReward = 0.0
    private var episodeLength = 0
    private val logFile = File(logPath).apply {
        if (!exists()) writeText("episode,reward,length\n")
    }

    /**
     * Resets the environment and associated counters to an initial state.
     * Logs the performance metrics of the completed episode before resetting, if applicable.
     *
     * @param seed An optional random seed for reproducibility. If `null`, no specific seed is used.
     * @param options An optional map of configuration options to customize the reset process. The specific keys
     *                and values depend on the environment's implementation.
     * @return The initial state of the environment, wrapped in an `InitialState` object containing the initial
     *         state and any relevant metadata.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        if (episodeLength > 0) {
            // Log to file beforeStep resetting counters (if previous episode finished)
            logFile.appendText("${System.currentTimeMillis()},$episodeReward,$episodeLength\n")
        }
        episodeReward = 0.0
        episodeLength = 0
        return env.reset(seed, options)
    }

    /**
     * Executes a single step in the monitored environment using the provided action.
     * Updates the episode metrics such as total reward and episode length.
     * If the episode ends due to termination or truncation, logs the metrics to a file
     * and optionally includes the episode statistics in the returned step information.
     *
     * @param action The action to be performed within the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and additional metadata, potentially augmented with episode statistics.
     */
    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        episodeReward += t.reward
        episodeLength += 1
        if (t.terminated || t.truncated) {
            // Log to file at end of episode
            logFile.appendText("${System.currentTimeMillis()},$episodeReward,$episodeLength\n")
            // Optionally, also attach stats to info:
            val newInfo = t.info.toMutableMap()
            newInfo["monitor"] = "reward=$episodeReward,length=$episodeLength"
            return t.copy(info = newInfo)
        }
        return t
    }
}
