package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import javafx.application.*
import javafx.scene.*
import javafx.scene.layout.*
import javafx.scene.media.*
import javafx.stage.*
import org.jcodec.api.awt.*
import org.jetbrains.kotlinx.jupyter.api.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import java.awt.*
import java.awt.image.*
import java.io.*

fun flattenObservation(obs: Any?, dtype: DataType): List<Number> = when (obs) {
    is Number -> listOf(obs)
    is Boolean -> listOf(if (obs) 1 else 0)
    is FloatArray -> obs.toList()
    is DoubleArray -> obs.toList()
    is IntArray -> obs.toList()
    is LongArray -> obs.toList()
    is BooleanArray -> obs.map { if (it) 1 else 0 }
    is NDArray<*, *> -> obs.data.map { it as Number }
    is List<*> -> obs.flatMap { flattenObservation(it, dtype) }
    is Map<*, *> -> obs.values.flatMap { flattenObservation(it, dtype) }
    is OneOfSample -> flattenObservation(obs.value, dtype)
    null -> emptyList()
    else -> throw IllegalArgumentException("Unsupported observation type: ${obs?.javaClass}")
}

fun <Num : Number> toNDArray(nums: List<Number>, dtype: DataType): NDArray<Num, D1> {
    val shape = intArrayOf(nums.size)
    @Suppress("UNCHECKED_CAST")
    return when (dtype) {
        DoubleDataType -> mk.ndarray<Double, D1>(nums.map { it.toDouble() }, shape) as NDArray<Num, D1>
        FloatDataType -> mk.ndarray<Float, D1>(nums.map { it.toFloat() }, shape) as NDArray<Num, D1>
        IntDataType -> mk.ndarray<Int, D1>(nums.map { it.toInt() }, shape) as NDArray<Num, D1>
        LongDataType -> mk.ndarray<Long, D1>(nums.map { it.toLong() }, shape) as NDArray<Num, D1>
        else -> throw IllegalArgumentException("Unsupported dtype: $dtype")
    }
}

fun <Num : Number, D : Dimension> rescale(
    x: NDArray<Num, D>,
    srcLow: NDArray<Num, D>, srcHigh: NDArray<Num, D>,
    tgtLow: NDArray<Num, D>, tgtHigh: NDArray<Num, D>,
    dim: D
): NDArray<Num, D> {
    val srcL = srcLow.data
    val srcH = srcHigh.data
    val tgtL = tgtLow.data
    val tgtH = tgtHigh.data
    val xs = x.data

    val result = Array(xs.size) { i ->
        val sL = srcL[i].toDouble()
        val sH = srcH[i].toDouble()
        val tL = tgtL[i].toDouble()
        val tH = tgtH[i].toDouble()
        val v = xs[i].toDouble()
        if (sH == sL) tL else tL + (v - sL) * (tH - tL) / (sH - sL)
    }
    @Suppress("UNCHECKED_CAST")
    return when (x.dtype) {
        DoubleDataType -> mk.ndarray(result.map { it }, x.shape, dim) as NDArray<Num, D>
        FloatDataType -> mk.ndarray(result.map { it.toFloat() }, x.shape, dim) as NDArray<Num, D>
        IntDataType -> mk.ndarray(result.map { it.toInt() }, x.shape, dim) as NDArray<Num, D>
        LongDataType -> mk.ndarray(result.map { it.toLong() }, x.shape, dim) as NDArray<Num, D>
        ShortDataType -> mk.ndarray(result.map { it.toInt().toShort() }, x.shape, dim) as NDArray<Num, D>
        ByteDataType -> mk.ndarray(result.map { it.toInt().toByte() }, x.shape, dim) as NDArray<Num, D>
        else -> throw IllegalArgumentException("Unsupported dtype: ${x.dtype}")
    }
}
fun <Num : Number, D : Dimension> clipToBox(
    x: NDArray<Num, D>,
    box: Box<Num, D>
): NDArray<Num, D> {
    val l = box.low.data
    val h = box.high.data
    val xs = x.data

    val result = Array(xs.size) { i ->
        val v = xs[i].toDouble()
        val lo = l[i].toDouble()
        val hi = h[i].toDouble()
        when {
            v < lo -> lo
            v > hi -> hi
            else -> v
        }
    }

    @Suppress("UNCHECKED_CAST")
    return when (x.dtype) {
        DoubleDataType -> mk.ndarray(result.map { it }, x.shape, x.dim) as NDArray<Num, D>
        FloatDataType -> mk.ndarray(result.map { it.toFloat() }, x.shape, x.dim) as NDArray<Num, D>
        IntDataType -> mk.ndarray(result.map { it.toInt() }, x.shape, x.dim) as NDArray<Num, D>
        LongDataType -> mk.ndarray(result.map { it.toLong() }, x.shape, x.dim) as NDArray<Num, D>
        ShortDataType -> mk.ndarray(result.map { it.toInt().toShort() }, x.shape, x.dim) as NDArray<Num, D>
        ByteDataType -> mk.ndarray(result.map { it.toInt().toByte() }, x.shape, x.dim) as NDArray<Num, D>
        else -> throw IllegalArgumentException("Unsupported dtype: ${x.dtype}")
    }
}

fun renderFrameToBufferedImage(frame: Rendering.RenderFrame): BufferedImage {
    val img = BufferedImage(frame.width, frame.height, BufferedImage.TYPE_INT_RGB)
    val bytes = frame.bytes
    var idx = 0
    for (y in 0 until frame.height) {
        for (x in 0 until frame.width) {
            val r = bytes[idx++].toInt() and 0xFF
            val g = bytes[idx++].toInt() and 0xFF
            val b = bytes[idx++].toInt() and 0xFF
            val rgb = (r shl 16) or (g shl 8) or b
            img.setRGB(x, y, rgb)
        }
    }
    return img
}

fun saveEpisodeAsMp4JCodec(frames: List<BufferedImage>, folder: String, episode: Int = 1, fps: Int = 30) {
    val mp4File = File(folder, "episode_$episode.mp4")
    mp4File.parentFile?.mkdirs()
    val encoder = AWTSequenceEncoder.createSequenceEncoder(mp4File, fps)
    frames.forEach { encoder.encodeImage(it) }
    encoder.finish()
}

fun displayVideo(frames: List<Rendering.RenderFrame>, folder: String): Any {
    saveEpisodeAsMp4JCodec(frames.map { renderFrameToBufferedImage(it) }, folder)
    return displayVideo(File(folder, "episode_1.mp4"), frames.first().width.toDouble(), frames.first().height.toDouble())
}

private object JavaFXState {
    @Volatile var launched = false
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
