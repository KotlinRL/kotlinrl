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
    private val fps: Int = 30,
    private val displayLastOnClose: Boolean = true,
) : SimpleWrapper<O, A, OS, AS>(env) {

    private var episodeCount = 0
    private var record = false
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

    override fun close() {
        super.close()
        if(displayLastOnClose) {
            displayVideo(File(folder, "episode_${episodeCount}.mp4"))
        }
    }
}
