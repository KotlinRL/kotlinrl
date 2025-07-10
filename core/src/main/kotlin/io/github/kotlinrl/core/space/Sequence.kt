package io.github.kotlinrl.core.space

import kotlin.random.*

class Sequence<T>(
    val space: Space<T>,
    val maxLength: Int,
    val seed: Int? = null
) : Space<List<T>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): List<T> {
        val length = random.nextInt(maxLength + 1)
        return List(length) { space.sample() }
    }

    override fun contains(value: Any?): Boolean =
        value is List<*> &&
                value.size <= maxLength &&
                value.all { it != null && space.contains(it) }
}
