package io.github.kotlinrl.core

import io.github.kotlinrl.core.wrapper.episodeFolderName
import io.github.kotlinrl.core.wrapper.renderFrameToBufferedImage
import javafx.animation.KeyFrame
import javafx.animation.PauseTransition
import javafx.animation.Timeline
import javafx.application.Application
import javafx.application.Platform
import javafx.event.EventHandler
import javafx.geometry.Insets
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.control.Slider
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.HBox
import javafx.scene.layout.StackPane
import javafx.scene.layout.VBox
import javafx.scene.shape.SVGPath
import javafx.stage.Stage
import javafx.util.Duration
import org.jetbrains.kotlinx.jupyter.api.HTML
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.awt.Desktop
import java.awt.image.BufferedImage
import java.io.File
import kotlin.collections.first
import kotlin.collections.map


/**
 * Represents a composite key consisting of a state and an action, both of which implement
 * the `Comparable` interface. This key is typically used in contexts where state-action pairs
 * need to be uniquely identified and compared, such as Q-tables or similar data structures
 * in reinforcement learning.
 *
 * @param State The type of the state, which must implement `Comparable`.
 * @param Action The type of the action, which must implement `Comparable`.
 *
 * Implements the `Comparable` interface, allowing state-action keys to be ordered
 * lexicographically. The comparison is performed by comparing the state first, followed
 * by the action if the states are equal.
 */
data class StateActionKey<State : Comparable<State>, Action : Comparable<Action>>(
    val state: State,
    val action: Action
) : Comparable<StateActionKey<State, Action>> {
    override fun compareTo(other: StateActionKey<State, Action>): Int {
        val stateCmp = state.compareTo(other.state)
        return if (stateCmp != 0) stateCmp else action.compareTo(other.action)
    }
}

/**
 * Represents a wrapper for a list of integers with the ability to define custom comparison logic.
 *
 * This class provides a structure to encapsulate a list of integers and implement comparison between
 * instances of the wrapper based on their respective integer lists. It extends the `Comparable` interface to
 * allow use in sorted collections or other contexts requiring total ordering.
 *
 * @constructor Creates a new instance of `ComparableIntList` with the provided list of integers.
 * @property data The underlying list of integers that this wrapper encapsulates.
 */
@JvmInline
value class ComparableIntList(val data: List<Int>) : Comparable<ComparableIntList> {
    override fun compareTo(other: ComparableIntList): Int {
        return data.zip(other.data).fold(0) { acc, (a, b) ->
            if (acc != 0) acc else a.compareTo(b)
        }
    }

    fun toNDArray(): NDArray<Int, DN> = mk.ndarray(data).asDNArray()

    override fun toString() = data.joinToString(",")
}


/**
 * Constructs a `StateActionKey` for the given state and action.
 * The state and action must either be `Comparable` or transformable into a comparable form.
 * For states of type `NDArray<Int, DN>` and actions of type `Int`,
 * the state is mapped to a `ComparableIntList`. For other cases, both
 * state and action must already be `Comparable`.
 *
 * @param State The type of the state, which must be `Comparable` or transformable to a comparable form.
 * @param Action The type of the action, which must be `Comparable` or transformable to a comparable form.
 * @param s The state to be used in constructing the key.
 * @param a The action to be used in constructing the key.
 * @return A `StateActionKey` instance encapsulating the given state and action.
 * @throws IllegalArgumentException If the state and action cannot be transformed into a comparable form.
 */
@Suppress("UNCHECKED_CAST")
internal fun <State, Action> stateActionKey(s: State, a: Action): StateActionKey<*, *> =
    when (s) {
        is NDArray<*, *> if a is Int -> StateActionKey(ComparableIntList((s as NDArray<Int, DN>).toList()), a)
        is Comparable<*> if a is Comparable<*> -> {
            StateActionKey(s as Comparable<Any>, a as Comparable<Any>)
        }

        else -> error("State ($s) and Action ($a) must be Comparable or mappable to a comparable form.")
    }

/**
 * Determines a comparable key for the given state.
 *
 * The function generates a key that can be compared across instances of the same or compatible types.
 * For states of type `NDArray`, the key is created by converting the array to a list of integers
 * and wrapping it in a `ComparableIntList`. For states that are already of a type implementing `Comparable`,
 * the state itself is returned. Any state not falling into these categories will result in an error.
 *
 * @param State The type representing the state from which a comparable key is derived.
 * @param s The state object for which a comparable key needs to be generated.
 * @return A comparable object representing a consistent key for the provided state.
 * @throws IllegalArgumentException If the provided state is neither comparable nor mappable to a comparable key.
 */
@Suppress("UNCHECKED_CAST")
internal fun <State> stateKey(s: State): Comparable<*> =
    when (s) {
        is NDArray<*, *> -> ComparableIntList((s as NDArray<Int, DN>).toList())
        is Comparable<*> -> s
        else -> error("State $s must be Comparable or mappable to a comparable key.")
    }

/**
 * Computes a comparable key for the given action.
 *
 * @param a The action for which a comparable key is required. Must either be of a type that implements `Comparable`
 *          or mappable to a comparable key.
 * @return A comparable key corresponding to the given action.
 * @throws IllegalStateException if the action does not implement `Comparable` and cannot be mapped to a comparable key.
 */
internal fun <Action> actionKey(a: Action): Comparable<*> =
    when (a) {
        is Comparable<*> -> a
        else -> error("Action $a must be Comparable or mappable to a comparable key.")
    }

/**
 * Displays a video located in the folder and episode specified.
 *
 * @param folder The path to the folder containing the video file.
 * @param episode The episode number used to retrieve the corresponding video folder name.
 * @return The rendered video output, which might vary based on the environment
 *         (e.g., HTML video element, JavaFX player, or fallback to the system's default video player).
 */
fun displayVideo(
    folder: String,
    episode: Int
): Any? {
    return displayVideo(File(folder, episodeFolderName(episode)))
}

/**
 * Displays a video from the specified file.
 *
 * This function accepts a file path as a string, processes it into a `File`
 * object, and attempts to display the video. The video is rendered differently
 * depending on the execution environment (e.g., Jupyter Notebook or standalone
 * application).
 *
 * @param file The path to the video file to be displayed as a string.
 * @return The result of the video display operation. In Jupyter Notebook environments,
 *         this will be an `HTML` object containing the video embed. In other contexts,
 *         the return value may vary depending on the display method or fallback mechanism.
 */
fun displayVideo(file: String): Any {
    return displayVideo(File(file))
}

/**
 * Displays a video file using appropriate rendering mechanisms based on the environment.
 * In a Jupyter Notebook, it outputs the video as an HTML element.
 * On other environments, it attempts to open the video using JavaFX or the default desktop player.
 *
 * @param file the video file to be displayed
 * @return an object representing the rendered video content or an empty string if the video is opened in an external player
 */
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

/**
 * Converts the current `RenderFrame` instance into a `BufferedImage`.
 * Provides a pixel-perfect graphical representation of the render frame.
 *
 * @return A `BufferedImage` generated from this `RenderFrame`.
 */
fun RenderFrame.toBufferedImage(): BufferedImage = renderFrameToBufferedImage(this)

/**
 * A singleton object serving as a state tracker for JavaFX-related initialization.
 *
 * This object maintains a single volatile Boolean property, `launched`, which is used
 * to indicate whether the JavaFX runtime environment has been successfully launched.
 * The volatile keyword ensures visibility of updates to this property across threads,
 * enabling safe concurrent access.
 *
 * This is particularly useful in scenarios where JavaFX initialization must only occur once
 * and further invocations can rely on this state to avoid redundant initialization or incorrect
 * behavior in concurrent environments.
 *
 * Properties:
 * - `launched`: A Boolean flag indicating the initialization state of JavaFX. Defaults to `false`.
 */
private object JavaFXState {
    @Volatile
    var launched = false
}

/**
 * A JavaFX application that plays a sequence of image frames as an animation. The application provides
 * controls for navigating through the frames, adjusting playback speed, and pausing or playing the animation.
 */
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

            val buttonRow =
                HBox(10.0, stepLeftButton, rewindButton, playPauseButton, forwardButton, stepRightButton).apply {
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

        fun svgButton(svgPath: String, size: Double = 24.0, fill: String = "white"): Button {
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
