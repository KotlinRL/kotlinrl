package io.github.kotlinrl.core.space

import kotlin.random.*

class MultiDiscrete(
    vararg val nvec: Int,
    val seed: Int? = null
) : Space<IntArray> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): IntArray =
        IntArray(nvec.size) { i -> random.nextInt(nvec[i]) }

    override fun contains(value: Any?): Boolean =
        value is IntArray &&
                value.size == nvec.size &&
                value.indices.all { value[it] in 0 until nvec[it] }
}
