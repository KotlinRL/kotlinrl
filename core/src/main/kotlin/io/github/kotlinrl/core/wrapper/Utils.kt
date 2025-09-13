package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jcodec.api.awt.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import java.awt.image.*
import java.io.*
import javax.imageio.*

/**
 * Flattens a nested observation structure into a list of numeric values.
 * This function supports various data types including primitive arrays,
 * lists, maps, and custom structures like `OneOfSample`.
 *
 * @param obs The observation to be flattened. It can be a single value, an array,
 * a list, a map, an `NDArray`, a boolean, or a custom structure like `OneOfSample`.
 * @param dtype The data type to interpret the flattened values.
 * @return A list of numeric values representing the flattened observation.
 * @throws IllegalArgumentException if the observation type is unsupported.
 */
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

/**
 * Converts a list of numbers into a 1-dimensional NDArray with the specified data type.
 *
 * @param nums The list of numbers to be converted.
 * @param dtype The data type of the resulting NDArray (e.g., Double, Float, Int, Long).
 * @return A 1-dimensional NDArray of the specified data type containing the provided numbers.
 * @throws IllegalArgumentException If the specified data type is not supported.
 */
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

/**
 * Rescales the values of an NDArray from a source range to a target range along the specified dimension.
 *
 * @param x The input NDArray whose values need to be rescaled.
 * @param srcLow An NDArray specifying the lower bounds of the source range.
 * @param srcHigh An NDArray specifying the upper bounds of the source range.
 * @param tgtLow An NDArray specifying the lower bounds of the target range.
 * @param tgtHigh An NDArray specifying the upper bounds of the target range.
 * @param dim The dimension along which the rescaling is applied.
 * @return An NDArray with values rescaled from the source range to the target range.
 */
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

/**
 * Clips the values of an NDArray to be within the bounds of a specified Box.
 * Each element in the NDArray is constrained to lie between the corresponding
 * low and high bounds defined by the Box.
 *
 * @param x The NDArray whose values are to be clipped. Each element will be checked
 *          and adjusted based on the bounds provided by the Box.
 * @param box A Box object defining the lower and upper bounds for each dimension.
 *            The bounds are applied element-wise to the NDArray.
 * @return A new NDArray with the same shape as the input NDArray, where every
 *         element is clipped to lie within the specified bounds of the Box.
 */
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

private val digits = 5
private val numberFormat = "%0${digits}d"

fun episodeFolderName(episode: Int) =
    "episode_${numberFormat.format(episode)}"

/**
 * Converts a given `RenderFrame` object into a `BufferedImage`.
 * The `RenderFrame` contains raw RGB byte data, which is processed and mapped to the pixels of the `BufferedImage`.
 *
 * @param frame The input render frame containing raw pixel data and dimensions for the image.
 * @return A BufferedImage representation of the input `RenderFrame`.
 */
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

/**
 * Saves a rendered frame as a PNG image file in a specified folder structure.
 *
 * @param frame The rendered frame to be saved. The frame contains raw pixel data to generate a PNG image.
 * @param folder The root directory under which the PNG image will be saved.
 * @param episode The episode number, used to generate the episode-specific subfolder.
 * @param frameIdx The frame index within the episode, used to name the generated PNG file.
 */
fun saveFrameAsPng(frame: Rendering.RenderFrame, folder: String, episode: Int, frameIdx: Int) {
    val img = renderFrameToBufferedImage(frame)
    val pngFile = File(folder, "${episodeFolderName(episode)}/frame_${numberFormat.format(frameIdx)}.png")
    pngFile.parentFile?.mkdirs()
    ImageIO.write(img, "png", pngFile)
}

/**
 * Deletes the specified file or directory recursively.
 * If the given file is a directory, all files and subdirectories
 * within it are deleted before deleting the directory itself.
 *
 * @param file The file or directory to delete. Must be a valid file or directory.
 */
fun deleteRecursively(file: File) {
    if (file.isDirectory) {
        file.listFiles()?.forEach(::deleteRecursively)
    }
    file.delete()
}

/**
 * Saves an episode as an MP4 file using JCodec, combining a sequence of PNG images into a video.
 *
 * @param folder The directory where the episode folder containing PNG images is located.
 * @param episode The episode number for which the MP4 file is created. The folder containing the images must match the naming convention for episodes.
 */
fun saveEpisodeAsMp4JCodec(folder: String, episode: Int) {
    val baseName = episodeFolderName(episode)
    val episodeFolder = File(folder, baseName)
    val pngFiles = episodeFolder
        .listFiles { it.extension == "png" }
        ?.sortedBy { it.name } ?: return

    if (pngFiles.isEmpty()) return

    val mp4File = File(folder, "$baseName.mp4")

    val encoder = AWTSequenceEncoder.createSequenceEncoder(mp4File, 30)
    pngFiles.forEach { file ->
        val img = ImageIO.read(file)
        encoder.encodeImage(img)
    }
    encoder.finish()
}
