package io.github.kotlinrl.core.space

import kotlin.random.Random

data class OneOfSample(
    val index: Int,
    val value: Any
)

class OneOf(
    val spaces: List<Space<*>>,
    val seed: Int? = null
) : Space<OneOfSample> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): OneOfSample {
        val which = random.nextInt(spaces.size)
        val value = spaces[which].sample()!!
        return OneOfSample(which, value)
    }

    override fun contains(value: Any?): Boolean =
        value is OneOfSample &&
                value.index in spaces.indices &&
                spaces[value.index].contains(value.value)
}
