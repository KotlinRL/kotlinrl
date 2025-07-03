package io.github.kotlinrl.core.space

import kotlin.random.*

class Discrete(
    val n: Int,
    val start: Int,
    val seed: Int? = null
) : Space<Int> {
    override val random: Random  = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): Int =
        start + random.nextInt(n)

    fun sample(mask: BooleanArray? = null): Int =
        mask?.let {
            require(mask.size == n) { "Mask size must match the number of discrete actions (n)." }
            val validIndices = mask.withIndex().filter { it.value }.map { it.index }
            require(validIndices.isNotEmpty()) { "Mask must allow at least one valid action." }
            start + validIndices[random.nextInt(validIndices.size)]
        } ?: sample()

    override fun contains(value: Any?): Boolean =
        value in start until (start + n)

    override fun toString(): String {
        return "Discrete(start=$start, n=$n, seed=$seed)"
    }
}