package io.github.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

class MultiBinary(
    val n: Int,
    val seed: Int? = null
) : Space<NDArray<Int, D1>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): NDArray<Int, D1> =
        mk.ndarray(IntArray(n) { random.nextInt(2) }) // 0 or 1

    override fun contains(value: Any?): Boolean =
        value is NDArray<*, *> &&
                value.dtype == DataType.IntDataType &&
                value.dim == D1 &&
                value.shape[0] == n &&
                value.indices.all { i ->
                    @Suppress("UNCHECKED_CAST")
                    val v = (value as NDArray<Int, D1>)[i]
                    v in 0..1
                }
}
