package org.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

class BoxNDArrayD1<T : Number>(
    val low: NDArray<T, D1>,
    val high: NDArray<T, D1>,
    val type: Class<T>,
    val seed: Int? = null
) : Space<NDArray<T, D1>> {
    override val random: Random = seed?.let { Random(it) } ?: Random.Default
    private val bounds = low.data.zip(high.data)
    val dtype: DataType = when (type) {
        Double::class.java -> DataType.DoubleDataType
        Float::class.java -> DataType.FloatDataType
        Int::class.java -> DataType.IntDataType
        Long::class.java -> DataType.LongDataType
        Short::class.java -> DataType.ShortDataType
        Byte::class.java -> DataType.ByteDataType
        else -> throw IllegalArgumentException("Invalid type parameter: $type. Only Number subclasses are supported.")
    }

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
    override fun sample(): NDArray<T, D1> {
        val sampledList: List<T> = bounds.map { (l, h) ->
            val lo = l.toDouble()
            val hi = h.toDouble()
            when (type) {
                Double::class.java -> {
                    val v = if (lo == hi) lo else random.nextDouble(lo, hi)
                    v.coerceIn(lo, hi)
                }

                Float::class.java -> {
                    val v = if (lo == hi) lo.toFloat() else random.nextDouble(lo, hi).toFloat()
                    v.coerceIn(lo.toFloat(), hi.toFloat())
                }

                Int::class.java -> {
                    if (lo == hi) lo.toInt() else random.nextInt(lo.toInt(), hi.toInt() + 1)
                }

                Long::class.java -> {
                    if (lo == hi) lo.toLong() else random.nextLong(lo.toLong(), hi.toLong() + 1)
                }

                Short::class.java -> {
                    if (lo == hi) lo.toInt().toShort() else random.nextInt(lo.toInt(), hi.toInt() + 1).toShort()
                }

                Byte::class.java -> {
                    if (lo == hi) lo.toInt().toByte() else random.nextInt(lo.toInt(), hi.toInt() + 1).toByte()
                }

                else -> throw UnsupportedOperationException("Unsupported data type.")
            } as T
        }
        return mk.ndarray(sampledList, shape = low.shape, dim = D1)
    }

    override fun contains(value: NDArray<T, D1>): Boolean {
        if (!value.shape.contentEquals(low.shape)) return false

        return value.data.zip(bounds) { v, (l, h) ->
            v.toDouble() in l.toDouble()..h.toDouble()
        }.all { it }
    }

    @Suppress("UNCHECKED_CAST") // Suppress this warning since we've controlled the type above
    private fun maxValueForType(): T = when (type) {
        Double::class.java -> Double.POSITIVE_INFINITY as T
        Float::class.java -> Float.POSITIVE_INFINITY as T
        Int::class.java -> Int.MAX_VALUE as T
        Long::class.java -> Long.MAX_VALUE as T
        Short::class.java -> Short.MAX_VALUE as T
        Byte::class.java -> Byte.MAX_VALUE as T
        else -> throw IllegalArgumentException("Unsupported type: $type")
    }

    @Suppress("UNCHECKED_CAST") // Suppress this warning since we've controlled the type above
    private fun minValueForType(): T = when (type) {
        Double::class.java -> Double.NEGATIVE_INFINITY as T
        Float::class.java -> Float.NEGATIVE_INFINITY as T
        Int::class.java -> Int.MIN_VALUE as T
        Long::class.java -> Long.MIN_VALUE as T
        Short::class.java -> Short.MIN_VALUE as T
        Byte::class.java -> Byte.MIN_VALUE as T
        else -> throw IllegalArgumentException("Unsupported type: $type")
    }
}