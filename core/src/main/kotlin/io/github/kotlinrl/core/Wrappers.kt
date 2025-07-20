package io.github.kotlinrl.core

import io.github.kotlinrl.core.wrapper.*
import javafx.animation.*
import javafx.application.*
import javafx.event.*
import javafx.geometry.*
import javafx.geometry.Insets
import javafx.scene.*
import javafx.scene.control.*
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.image.*
import javafx.scene.image.Image
import javafx.scene.input.*
import javafx.scene.layout.*
import javafx.scene.shape.*
import javafx.stage.*
import javafx.util.*
import org.jetbrains.kotlinx.jupyter.api.*
import java.awt.*
import java.awt.image.BufferedImage
import java.io.*

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
    folder: String
): Any? {
    val digits = 5
    val numberFormat = "%0${digits}"
    return displayVideo(File(folder, "episode_${numberFormat.format(episode)}"))
}

fun displayVideo(file: File): Any {
    // Try notebook HTML
    return if (System.getenv("JPY_PARENT_PID") != null) {
        val cwd = File(".").absoluteFile.normalize()
        val absPath = file.absoluteFile
        val relPath = absPath.relativeToOrNull(cwd)?.path ?: file.name

        HTML(
            """<video style="max-width: 100%; height: auto;" controls>
          <source src="${relPath}" type="video/mp4">
          Your browser does not support the video tag.
        </video>"""
        )
    } else {
        try {
            if (!JavaFXState.launched) {
                JavaFXState.launched = true
                Thread {
                    Application.launch(FramePlayer::class.java)
                }.start()
                Thread.sleep(500)
            }
            Platform.runLater {
                FramePlayer.play(file)
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

fun RenderFrame.toBufferedImage(): BufferedImage = renderFrameToBufferedImage(this)

private object JavaFXState {
    @Volatile
    var launched = false
}

class FramePlayer : Application() {
    override fun start(stage: Stage) {

    }

    companion object {
        fun formatFrameName(index: Int, frames: List<File>): String {
            return frames.getOrNull(index)?.name ?: "frame_%04d.png".format(index)
        }

        fun loadFrameFiles(folder: File): List<File> {
            return folder.listFiles { f -> f.name.endsWith(".png") }
                ?.sortedBy { it.name.removePrefix("frame_").removeSuffix(".png").toIntOrNull() ?: Int.MAX_VALUE }
                ?: emptyList()
        }

        fun play(folder: File, initialFps: Int = 30) {
            val frameFiles = loadFrameFiles(folder)
            val frames = frameFiles.map { Image(it.toURI().toString()) }
            if (frames.isEmpty()) throw IllegalArgumentException("No frames found in $folder")

            val width = frames.first().width
            val height = frames.first().height
            val imageView = ImageView(frames.first()).apply {
                fitWidth = width
                fitHeight = height
                isPreserveRatio = true
            }

            val playIcon = "M8 5v14l11-7z"
            val pauseIcon = "M6 19h4V5H6v14zm8-14v14h4V5h-4z"
            val stepLeftButton = svgButton("M15 18V6l-6 6 6 6z")
            val stepRightButton = svgButton("M9 6v12l6-6-6-6z")
            val playPauseButton = svgButton(playIcon)
            val rewindButton = svgButton("M13 6v12l-8.5-6L13 6zm9 0v12l-8.5-6L22 6z")
            val forwardButton = svgButton("M4 6v12l8.5-6L4 6zm9 0v12l8.5-6L13 6z")
            val timeLabel = Label(formatFrameName(0, frameFiles)).apply {
                style = "-fx-text-fill: white; -fx-font-weight: bold;"
            }
            val timeSlider = Slider(0.0, frames.size - 1.0, 0.0)
            val fpsSlider = Slider(6.0, 60.0, initialFps.toDouble()).apply {
                majorTickUnit = 15.0
                minorTickCount = 0
                isSnapToTicks = true
                isShowTickMarks = true
                isShowTickLabels = true
                prefWidth = 100.0
            }
            val fpsLabel = Label("${initialFps} fps")

            var currentFrame = 0
            var isPlaying = false

            val timeline = Timeline().apply {
                cycleCount = Timeline.INDEFINITE
            }

            fun updateTimelineRate(fps: Double) {
                timeline.stop()
                timeline.keyFrames.clear()
                timeline.keyFrames.add(
                    KeyFrame(Duration.millis(1000.0 / fps), EventHandler {
                        if (isPlaying) {
                            if (currentFrame < frames.size - 1) {
                                currentFrame++
                                imageView.image = frames[currentFrame]
                                timeSlider.value = currentFrame.toDouble()
                                timeLabel.text = formatFrameName(currentFrame, frameFiles)
                            } else {
                                isPlaying = false
                                (playPauseButton.graphic as SVGPath).content = playIcon
                            }
                        }
                    })
                )
                timeline.play()
            }

            fpsSlider.valueProperty().addListener { _, _, newVal ->
                updateTimelineRate(newVal.toDouble())
                fpsLabel.text = "${newVal.toInt()} fps"
            }

            updateTimelineRate(initialFps.toDouble())

            timeSlider.valueChangingProperty().addListener { _, _, changing ->
                if (!changing) {
                    currentFrame = timeSlider.value.toInt().coerceIn(0, frames.size - 1)
                    imageView.image = frames[currentFrame]
                    timeLabel.text = formatFrameName(currentFrame, frameFiles)
                }
            }

            playPauseButton.setOnAction {
                isPlaying = !isPlaying
                (playPauseButton.graphic as SVGPath).content = if (isPlaying) pauseIcon else playIcon
            }

            val availableFps = listOf(6.0, 15.0, 30.0, 45.0, 60.0)
            rewindButton.setOnAction {
                fpsSlider.value = availableFps.filter { it < fpsSlider.value }.maxOrNull() ?: 15.0
            }

            forwardButton.setOnAction {
                fpsSlider.value = availableFps.filter { it > fpsSlider.value }.minOrNull() ?: 60.0
            }

            stepLeftButton.setOnAction {
                isPlaying = false
                (playPauseButton.graphic as SVGPath).content = playIcon
                currentFrame = (currentFrame - 1).coerceAtLeast(0)
                imageView.image = frames[currentFrame]
                timeSlider.value = currentFrame.toDouble()
                timeLabel.text = formatFrameName(currentFrame, frameFiles)
            }

            stepRightButton.setOnAction {
                isPlaying = false
                (playPauseButton.graphic as SVGPath).content = playIcon
                currentFrame = (currentFrame + 1).coerceAtMost(frames.size - 1)
                imageView.image = frames[currentFrame]
                timeSlider.value = currentFrame.toDouble()
                timeLabel.text = formatFrameName(currentFrame, frameFiles)
            }

            val buttonRow = HBox(10.0, stepLeftButton, rewindButton, playPauseButton, forwardButton, stepRightButton).apply {
                alignment = Pos.CENTER
            }

            val labelRow = HBox(timeLabel).apply {
                alignment = Pos.CENTER
            }

            val controlRow = VBox(5.0, buttonRow, labelRow).apply {
                alignment = Pos.CENTER
                padding = Insets(10.0)
                style = "-fx-background-color: rgba(0,0,0,0.6); -fx-background-radius: 10;"
                isVisible = false
            }

            val sliderRow = HBox(10.0, fpsSlider, fpsLabel, timeSlider).apply {
                alignment = Pos.TOP_CENTER
                padding = Insets(10.0)
                style = "-fx-background-color: rgba(0,0,0,0.3); -fx-background-radius: 10; -fx-background-radius: 10;"
                isVisible = false
            }

            val root = StackPane(imageView, VBox(controlRow, sliderRow).apply {
                alignment = Pos.BOTTOM_CENTER
                padding = Insets(20.0)
            })
            fun hideControlsIfNotHovering() {
                PauseTransition(Duration.millis(150.0)).apply {
                    setOnFinished {
                        if (!controlRow.isHover && !sliderRow.isHover && !root.isHover) {
                            controlRow.isVisible = false
                            sliderRow.isVisible = false
                        }
                    }
                    play()
                }
            }

            fun showControls() {
                controlRow.isVisible = true
                sliderRow.isVisible = true
            }

            root.setOnMouseMoved { showControls() }
            root.setOnMouseExited { hideControlsIfNotHovering() }

            controlRow.setOnMouseExited { hideControlsIfNotHovering() }
            sliderRow.setOnMouseExited { hideControlsIfNotHovering() }

            val scene = Scene(root, width, height)
            val finalStage = Stage()
            finalStage.title = folder.name
            finalStage.scene = scene
            finalStage.show()
        }

        fun svgButton(svgPath: String, size: Double = 24.0, fill: String = "white" ): Button {
            val icon = SVGPath().apply {
                content = svgPath
                style = "-fx-fill: $fill;"
                scaleX = size / 24
                scaleY = size / 24
            }
            return Button("", icon).apply {
                style = """
                    -fx-background-color: transparent;
                    -fx-padding: 6;
                    -fx-cursor: hand;
                """.trimIndent()
            }
        }
    }
}
