package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.awt.image.*
import java.io.File

class RecordVideo<
        O, A, OS : Space<O>, AS : Space<A>
        >(
    env: Env<O, A, OS, AS>,
    private val folder: String = "videos",
    private val every: Int = 1,
    private val fps: Int = 30
) : SimpleWrapper<O, A, OS, AS>(env) {

    private var episodeCount = 0
    private var record = false
    var width = 640.0
        private set
    var height = 480.0
        private set
    private val frames = mutableListOf<BufferedImage>()

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<O> {
        // If previous episode was recorded, save video
        if (frames.isNotEmpty() && record) {
            saveEpisodeAsMp4JCodec(frames, folder, episodeCount - 1, fps)
            frames.clear()
        }
        episodeCount++
        record = (episodeCount % every == 0)
        val initial = env.reset(seed, options)
        if (record) {
            val rendering = env.render()
            if (rendering is Rendering.RenderFrame) {
                width = rendering.width.toDouble()
                height = rendering.height.toDouble()
                frames.add(renderFrameToBufferedImage(rendering))
            }
        }
        return initial
    }

    override fun step(act: A): Transition<O> {
        val t = env.step(act)
        if (record) {
            val rendering = env.render()
            if (rendering is Rendering.RenderFrame) {
                frames.add(renderFrameToBufferedImage(rendering))
            }
        }
        if ((t.terminated || t.truncated) && record) {
            saveEpisodeAsMp4JCodec(frames, folder, episodeCount, fps)
            frames.clear()
        }
        return t
    }

    fun displayVideo(episode: Int): Any? {
        return displayVideo(File(folder, "episode_$episode.mp4"), width, height)
    }
}
