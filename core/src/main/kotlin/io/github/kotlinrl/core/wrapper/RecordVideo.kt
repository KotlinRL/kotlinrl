package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.awt.image.*

class RecordVideo<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    val folder: String = "videos",
    private val every: Int = 1
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    private var episodeCount = 0
    private var record = false
    private val frames = mutableListOf<BufferedImage>()

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<State> {
        episodeCount++
        record = (episodeCount % every == 0)
        val initial = env.reset(seed, options)
        if (record) {
            val rendering = env.render()
            if (rendering is Rendering.RenderFrame) {
                frames.add(renderFrameToBufferedImage(rendering))
            }
        }
        return initial
    }

    override fun step(action: Action): StepResult<State> {
        val t = env.step(action)
        if (record) {
            val rendering = env.render()
            if (rendering is Rendering.RenderFrame) {
                frames.add(renderFrameToBufferedImage(rendering))
            }
        }
        if ((t.terminated || t.truncated) && record) {
            saveEpisodeAsMp4JCodec(frames, folder, episodeCount)
            frames.clear()
        }
        return t
    }
}
