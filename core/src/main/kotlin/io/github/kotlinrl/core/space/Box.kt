package io.github.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.random.*

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

    override fun contains(value: Any?): Boolean {
        if (value !is NDArray<*, *>) return false
        if (!value.shape.contentEquals(low.shape)) return false

        return value.data.zip(bounds) { v, (l, h) ->
            (v as Number).toDouble() in l.toDouble()..h.toDouble()
        }.all { it }
    }

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