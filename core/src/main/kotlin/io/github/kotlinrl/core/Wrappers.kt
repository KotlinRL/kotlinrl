package io.github.kotlinrl.core

import io.github.kotlinrl.core.space.Space
import io.github.kotlinrl.core.wrapper.renderFrameToBufferedImage
import io.github.kotlinrl.core.wrapper.saveEpisodeAsMp4JCodec
import javafx.application.Application
import javafx.application.Platform
import javafx.scene.Scene
import javafx.scene.layout.StackPane
import javafx.scene.media.Media
import javafx.scene.media.MediaPlayer
import javafx.scene.media.MediaView
import javafx.stage.Stage
import org.jcodec.api.awt.AWTSequenceEncoder
import org.jetbrains.kotlinx.jupyter.api.HTML
import java.awt.Desktop
import java.awt.image.BufferedImage
import java.io.File
import kotlin.collections.forEach

typealias ClipAction<State, Num, D, ObservationSpace> = io.github.kotlinrl.core.wrapper.ClipAction<State, Num, D, ObservationSpace>
typealias FilterAction<State, ObservationSpace> = io.github.kotlinrl.core.wrapper.FilterAction<State, ObservationSpace>
typealias FilterObservation<Action, ActionSpace> = io.github.kotlinrl.core.wrapper.FilterObservation<Action, ActionSpace>
typealias FlattenObservation<Num, WrappedState, Action, WrappedObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.FlattenObservation<Num, WrappedState, Action, WrappedObservationSpace, ActionSpace>
typealias Monitor<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.Monitor<State, Action, ObservationSpace, ActionSpace>
typealias NormalizeReward<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.NormalizeReward<State, Action, ObservationSpace, ActionSpace>
typealias NormalizeState<Num, D, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.NormalizeState<Num, D, Action, ObservationSpace, ActionSpace>
typealias OrderEnforcing<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.OrderEnforcing<State, Action, ObservationSpace, ActionSpace>
typealias RecordEpisodeStatistics<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RecordEpisodeStatistics<State, Action, ObservationSpace, ActionSpace>
typealias RecordVideo<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RecordVideo<State, Action, ObservationSpace, ActionSpace>
typealias RescaleAction<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RescaleAction<State, Action, ObservationSpace, ActionSpace>
typealias RunningStats = io.github.kotlinrl.core.wrapper.RunningStats
typealias TimeLimit<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.TimeLimit<State, Action, ObservationSpace, ActionSpace>
typealias TransformReward<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.TransformReward<State, Action, ObservationSpace, ActionSpace>
typealias TransformState<State, Action, ObservationSpace, ActionSpace, WrappedState, WrappedObservationSpace> = io.github.kotlinrl.core.wrapper.TransformState<State, Action, ObservationSpace, ActionSpace, WrappedState, WrappedObservationSpace>

fun displayVideo(
    episode: Int,
    folder: String,
    width: Double = 640.0,
    height: Double = 480.0,
): Any? {
    return displayVideo(File(folder, "episode_$episode.mp4"), width, height)
}

fun displayVideo(frames: List<RenderFrame>, folder: String): Any {
    saveEpisodeAsMp4JCodec(frames.map { renderFrameToBufferedImage(it) }, folder)
    return displayVideo(File(folder, "episode_1.mp4"), frames.first().width.toDouble(), frames.first().height.toDouble())
}

fun displayVideo(file: File, width: Double = 640.0, height: Double = 480.0): Any {
    // Try notebook HTML
    return if (System.getenv("JPY_PARENT_PID") != null) {
        val cwd = File(".").absoluteFile.normalize()
        val absPath = file.absoluteFile
        val relPath = absPath.relativeToOrNull(cwd)?.path ?: file.name

        HTML("""<video width="$width" height="$height" controls>
          <source src="${relPath}" type="video/mp4">
          Your browser does not support the video tag.
        </video>""")
    } else {
        try {
            if (!JavaFXState.launched) {
                JavaFXState.launched = true
                Application.launch(Mp4Player::class.java, file.absolutePath, width.toString(), height.toString())
            } else {
                Platform.runLater {
                    Mp4Player.play(file, width, height)
                }
            }
        } catch (e: Throwable) {
            // Fallback
            if (Desktop.isDesktopSupported()) {
                Desktop.getDesktop().open(file)
            } else {
                println("MP4 saved to: ${file.absolutePath}")
                println("Please open it with your video player.")
            }
        }
        ""
    }
}

private object JavaFXState {
    @Volatile var launched = false
}

class Mp4Player : Application() {
    override fun start(stage: Stage) {
        val params = parameters.raw
        val mp4Path = params[0]
        val width = params.getOrNull(1)?.toDoubleOrNull() ?: 640.0
        val height = params.getOrNull(2)?.toDoubleOrNull() ?: 480.0

        play(File(mp4Path), width, height, stage)
    }

    companion object {
        fun play(file: File, width: Double, height: Double, stage: Stage? = null) {
            val media = Media(file.toURI().toString())
            val mediaPlayer = MediaPlayer(media)
            val mediaView = MediaView(mediaPlayer)
            mediaView.fitWidth = width
            mediaView.fitHeight = height

            val root = StackPane(mediaView)
            val scene = Scene(root, width, height)

            val finalStage = stage ?: Stage()
            finalStage.scene = scene
            finalStage.title = "Env Rendering: ${file.name}"
            finalStage.show()
            mediaPlayer.play()
        }
    }
}
