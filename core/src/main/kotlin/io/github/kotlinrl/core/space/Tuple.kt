package io.github.kotlinrl.core.space

import kotlin.random.*

class Tuple(
    val spaces: List<Space<Any>> = emptyList(),
    val seed: Int? = null
) : Space<List<Any>> {
    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): List<Any> {
        return spaces.map { it.sample() }
    }

    override fun contains(value: Any?): Boolean {
        if (value !is List<*>) return false
        if (value.size != spaces.size) return false
        return value.zip(spaces).all { (v, s) -> s.contains(v) }
    }
}
