package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*
import java.io.*

/**
 * Wraps an existing environment to enable video recording of episodes.
 *
 * This class allows capturing frames from the environment's rendering capability
 * and saving them as video files over specific episodes. The recorded videos
 * are saved in the specified folder at designated intervals.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space specifying the observation space for the state.
 *                         This space represents the constraints and structure of the state observations.
 * @param ActionSpace The type of space specifying the allowable actions in the environment.
 *                    This space represents the constraints and structure of the possible actions.
 * @param env The environment to be wrapped and monitored for video recording.
 *            It serves as the underlying environment whose behavior is observed.
 * @param folder The directory where recorded videos and frames are saved.
 *               Defaults to "videos".
 * @param every Determines the interval of episodes for which video recording occurs.
 *              For example, if the value is 1, every episode will be recorded.
 */
class RecordVideo<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    val folder: String = "videos",
    private val every: Int = 1,
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var episodeCount = 0
    private var record = false
    private var frameCount = 0

    init {
        val file = File(folder)
        deleteRecursively(file)
        file.mkdirs()
    }

    /**
     * Resets the environment and handles video recording initiation if applicable.
     *
     * This method overrides the base reset functionality to include video recording logic
     * every specified number of episodes. It increments the episode count, determines
     * whether video recording should occur, and captures an initial frame if recording is enabled.
     *
     * @param seed An optional random seed for reproducing deterministic behavior during the reset.
     *             If `null`, the environment will use its default random generator.
     * @param options A map of additional configuration options to influence the reset process.
     *                The specific keys and values depend on the environment's implementation.
     * @return The initial state of the environment after being reset, including its state
     *         and associated metadata encapsulated in an `InitialState`.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<State> {
        episodeCount++
        record = (episodeCount % every == 0)
        val initial = env.reset(seed, options)
        if (record) {
            frameCount = 0
            maybeCaptureFrame()
        }
        return initial
    }

    /**
     * Executes a single step in the environment with the specified action, including
     * handling video recording and saving the episode as a video file when applicable.
     *
     * The function updates the environment state by performing the given action and
     * may capture frames for recording, saving the episode as an MP4 file if the
     * episode ends and recording is enabled.
     *
     * @param action The action to be performed in the environment.
     * @return A StepResult containing the updated state, reward, termination status,
     * truncation status, and any additional metadata after the action is applied.
     */
    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        maybeCaptureFrame()
        if ((t.terminated || t.truncated) && record) {
            saveEpisodeAsMp4JCodec(folder, episodeCount)
        }
        return t
    }

    /**
     * Captures and saves a rendering frame as a PNG image if recording is enabled.
     *
     * This method evaluates whether recording is active. If so, it requests a rendering
     * of the current environment state and checks if the rendered result is a valid frame.
     * If a valid frame is produced, the frame is saved as a PNG image in the specified folder.
     * The frame file is named based on the current episode count and frame index, and the
     * frame index is incremented after each successful save.
     *
     * The rendering operation leverages the environment's `render` method, which might return
     * an empty rendering or a valid `RenderFrame`. Saving the frame as PNG is handled by
     * the `saveFrameAsPng` helper function.
     *
     * The recording behavior depends on the `record` flag, which determines whether
     * frames are captured and saved.
     */
    private fun maybeCaptureFrame() {
        if (!record) return
        val rendering = env.render()
        if (rendering is RenderFrame) {
            saveFrameAsPng(rendering, folder, episodeCount, frameCount++)
        }
    }
}
