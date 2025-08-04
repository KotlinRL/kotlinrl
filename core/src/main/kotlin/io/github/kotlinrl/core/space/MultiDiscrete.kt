package io.github.kotlinrl.core.space

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.*

class MultiDiscrete(
    vararg val nvec: Int,
    val seed: Int? = null
) : Space<NDArray<Int, D1>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): NDArray<Int, D1> =
        mk.ndarray(IntArray(nvec.size) { random.nextInt(nvec[it]) })

    override fun contains(value: Any?): Boolean =
        value is NDArray<*, *> &&
                value.dtype == DataType.IntDataType &&
                value.dim == D1 &&
                value.shape[0] == nvec.size &&
                @Suppress("UNCHECKED_CAST")
                (value as NDArray<Int, D1>).indices.all {
                    value[it] in 0 until nvec[it]
                }
}
