package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.io.*

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

    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        maybeCaptureFrame()
        if ((t.terminated || t.truncated) && record) {
            saveEpisodeAsMp4JCodec(folder, episodeCount)
        }
        return t
    }

    private fun maybeCaptureFrame() {
        if (!record) return
        val rendering = env.render()
        if (rendering is Rendering.RenderFrame) {
            saveFrameAsPng(rendering, folder, episodeCount, frameCount++)
        }
    }
}
