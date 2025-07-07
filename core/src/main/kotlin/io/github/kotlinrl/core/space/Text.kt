package io.github.kotlinrl.core.space

import kotlin.random.Random

class Text(
    val maxLength: Int,
    val charset: Set<Char> = (' '..'~').toSet(),
    val seed: Int? = null
) : Space<String> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): String {
        val length = random.nextInt(maxLength + 1)
        return (1..length)
            .map { charset.random(random) }
            .joinToString("")
    }

    override fun contains(value: Any?): Boolean =
        value is String &&
                value.length <= maxLength &&
                value.all { it in charset }
}
