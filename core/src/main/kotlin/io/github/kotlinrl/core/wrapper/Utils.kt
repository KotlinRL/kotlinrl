package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.RenderFrame
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
import javax.imageio.ImageIO

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
    else -> throw IllegalArgumentException("Unsupported state type: ${obs?.javaClass}")
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

fun renderFrameToBufferedImage(frame: RenderFrame): BufferedImage {
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

fun saveEpisodeAsMp4JCodec(frames: List<BufferedImage>, folder: String, episode: Int = 1) {

    fun deleteRecursively(file: File) {
        if (file.isDirectory) {
            file.listFiles()?.forEach(::deleteRecursively)
        }
        file.delete()
    }

    val mp4File = File(folder, "episode_$episode.mp4")
    mp4File.parentFile?.mkdirs()
    val encoder = AWTSequenceEncoder.createSequenceEncoder(mp4File, 30)
    frames.forEach { encoder.encodeImage(it) }
    encoder.finish()

    val pngFolder = File(folder, "episode_$episode")
    if (pngFolder.exists()) {
        deleteRecursively(pngFolder)
    }
    pngFolder.mkdirs()

    val digits = frames.size.toString().length
    val numberFormat = "frame_%0${digits}d.png"
    frames.forEachIndexed { idx, img ->
        val pngFile = File(pngFolder, numberFormat.format(idx + 1))
        ImageIO.write(img, "png", pngFile)
    }
}