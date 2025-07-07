package io.github.kotlinrl.core.space

import kotlin.random.Random

class MultiBinary(
    val n: Int,
    val seed: Int? = null
) : Space<IntArray> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): IntArray =
        IntArray(n) { random.nextInt(2) } // 0 or 1

    override fun contains(value: Any?): Boolean =
        value is IntArray && value.size == n && value.all { it == 0 || it == 1 }
}
