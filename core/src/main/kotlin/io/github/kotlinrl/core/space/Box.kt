package io.github.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.random.*

/**
 * Represents a bounded space defined by low and high values for each dimension. This class generates
 * samples of a specific numeric type within the defined bounds and allows checking whether a value
 * exists within the space.
 *
 * @param T The numeric type of the values (e.g., Int, Float, Double, etc.), constrained to subclasses of `Number`.
 * @param D The dimensionality of the space, constrained to `Dimension`.
 * @property low An `NDArray` representing the lower bounds of the space across all dimensions.
 * @property high An `NDArray` representing the upper bounds of the space across all dimensions.
 * @property dtype The data type used for the space, which influences the kind of data sampled and validated.
 * @property seed An optional integer seed for the random number generator, enabling deterministic sampling.
 */
class Box<T : Number, D : Dimension>(
    val low: NDArray<T, D>,
    val high: NDArray<T, D>,
    val dtype: DataType,
    val seed: Int? = null
) : Space<NDArray<T, D>> {
    override val random: Random = seed?.let { Random(it) } ?: Random.Default
    private val bounds = low.data.zip(high.data)

    init {
        require(low.shape.contentEquals(high.shape)) {
            "Low and high bounds must have the same shape. Found low=${low.shape} and high=${high.shape}"
        }
        require(low.data.none { it == maxValueForType() }) {
            "No low value can be equal to the maximum possible value for the type, low=${low.data.toList()}"
        }
        require(high.data.none { it == minValueForType() }) {
            "No high value can be equal to the minimum possible value for the type, high=${high.data.toList()}"
        }
        require(
            bounds.all { (l, h) ->
                val lo = l.toDouble()
                val hi = h.toDouble()
                when {
                    lo.isNaN() && hi.isNaN() -> true   // no bounds
                    lo.isNaN() || hi.isNaN() -> false  // only one NaN? invalid
                    lo.isInfinite() && hi.isInfinite() -> true
                    lo <= hi -> true
                    else -> false
                }
            }
        ) {
            "Each low value must be less than or equal to its corresponding high value."
        }
        require(bounds.isNotEmpty()) { "Low and high arrays must not be empty." }

    }

    /**
     * Generates a sampled NDArray of type T and dimensionality D, based on specified bounds and data type.
     * Each element in the NDArray is sampled independently according to the range defined by bounds.
     * The sampling behavior is determined by the dtype property, ensuring that sampled elements
     * are coerced within their corresponding bounds.
     *
     * @return An NDArray of sampled values with the same shape and dimensionality as the lower bound (low).
     * The elements are of type T, generated based on the specified dtype and bounds.
     */
    @Suppress("UNCHECKED_CAST")
    override fun sample(): NDArray<T, D> {
        val sampledList: List<T> = bounds.map { (l, h) ->
            val lo = l.toDouble()
            val hi = h.toDouble()
            when (dtype) {
                DoubleDataType -> {
                    val v = if (lo == hi) lo else random.nextDouble(lo, hi)
                    v.coerceIn(lo, hi)
                }

                FloatDataType -> {
                    val v = if (lo == hi) lo.toFloat() else random.nextDouble(lo, hi).toFloat()
                    v.coerceIn(lo.toFloat(), hi.toFloat())
                }

                IntDataType -> {
                    if (lo == hi) lo.toInt() else random.nextInt(lo.toInt(), hi.toInt() + 1)
                }

                LongDataType -> {
                    if (lo == hi) lo.toLong() else random.nextLong(lo.toLong(), hi.toLong() + 1)
                }

                ShortDataType -> {
                    if (lo == hi) lo.toInt().toShort() else random.nextInt(lo.toInt(), hi.toInt() + 1).toShort()
                }

                ByteDataType -> {
                    if (lo == hi) lo.toInt().toByte() else random.nextInt(lo.toInt(), hi.toInt() + 1).toByte()
                }

                else -> throw UnsupportedOperationException("Unsupported data type.")
            } as T
        }
        return mk.ndarray(elements = sampledList, shape = low.shape, dim = low.dim)
    }

    /**
     * Checks if a given value is contained within the bounds defined by this Box.
     * The value must be an NDArray with a shape matching the lower bound (low).
     * Each element of the NDArray must lie within the corresponding range defined
     * by the bounds for this Box.
     *
     * @param value The value to check. It should be an NDArray matching the shape
     *              of `low` and each of its elements must lie within the specified bounds.
     * @return `true` if the value is an NDArray with a matching shape and whose
     *         elements are within the defined bounds; `false` otherwise.
     */
    override fun contains(value: Any?): Boolean {
        if (value !is NDArray<*, *>) return false
        if (!value.shape.contentEquals(low.shape)) return false

        return value.data.zip(bounds) { v, (l, h) ->
            (v as Number).toDouble() in l.toDouble()..h.toDouble()
        }.all { it }
    }

    /**
     * Determines the maximum representable value for the specified data type.
     *
     * @return The maximum value of type `T` corresponding to the current `dtype`.
     * @throws IllegalArgumentException if the `dtype` is not a supported type.
     */
    @Suppress("UNCHECKED_CAST") // Suppress this warning since we've controlled the type above
    private fun maxValueForType(): T = when (dtype) {
        DoubleDataType -> Double.POSITIVE_INFINITY as T
        FloatDataType -> Float.POSITIVE_INFINITY as T
        IntDataType -> Int.MAX_VALUE as T
        LongDataType -> Long.MAX_VALUE as T
        ShortDataType -> Short.MAX_VALUE as T
        ByteDataType -> Byte.MAX_VALUE as T
        else -> throw IllegalArgumentException("Unsupported type: $dtype")
    }

    /**
     * Determines the minimum representable value for the specified data type.
     *
     * @return The minimum value of type `T` corresponding to the current `dtype`.
     * @throws IllegalArgumentException if the `dtype` is not a supported type.
     */
    @Suppress("UNCHECKED_CAST") // Suppress this warning since we've controlled the type above
    private fun minValueForType(): T = when (dtype) {
        DoubleDataType -> Double.NEGATIVE_INFINITY as T
        FloatDataType -> Float.NEGATIVE_INFINITY as T
        IntDataType -> Int.MIN_VALUE as T
        LongDataType -> Long.MIN_VALUE as T
        ShortDataType -> Short.MIN_VALUE as T
        ByteDataType -> Byte.MIN_VALUE as T
        else -> throw IllegalArgumentException("Unsupported type: $dtype")
    }
}