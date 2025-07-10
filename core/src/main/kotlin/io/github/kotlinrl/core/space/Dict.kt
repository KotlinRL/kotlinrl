package io.github.kotlinrl.core.space

import kotlin.random.*

class Dict(
    val spaces: Map<String, Space<*>>,
    val seed: Int? = null
) : Space<Map<String, Any>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): Map<String, Any> =
        spaces.mapValues { (_, space) -> space.sample() as Any }

    override fun contains(value: Any?): Boolean {
        if (value !is Map<*, *>) return false
        if (value.size != spaces.size) return false

        return spaces.all { (key, space) ->
            value.containsKey(key) && space.contains(value[key])
        }
    }
}
