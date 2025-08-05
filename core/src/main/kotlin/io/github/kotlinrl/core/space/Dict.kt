package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * Represents a dictionary-like space, where each key-value pair is defined by independent subspaces.
 * This class is a composite space that aggregates multiple `Space` instances under a map-like structure.
 *
 * @property spaces A map representing the subspaces, where the key is a string identifier and the value
 * is a `Space` instance. Each subspace defines its own sampling and containment logic for values under the key.
 * @property seed An optional seed for the random number generator to ensure deterministic sampling.
 * If no seed is provided, the default random generator is used.
 */
class Dict(
    val spaces: Map<String, Space<*>>,
    val seed: Int? = null
) : Space<Map<String, Any>> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    override fun sample(): Map<String, Any> =
        spaces.mapValues { (_, space) -> space.sample() as Any }

    /**
     * Checks whether the given value is contained within the composite space defined by this instance.
     * The value must be a map where each key matches a key in the `spaces` map, and its corresponding value
     * must also be contained within the associated `Space` of the same key.
     *
     * @param value The value to check, which should be a map with keys and values matching the subspaces.
     * @return `true` if the value is a map, has the same size as `spaces`, and each key-value pair satisfies
     * the requirements of the corresponding subspace; `false` otherwise.
     */
    override fun contains(value: Any?): Boolean {
        if (value !is Map<*, *>) return false
        if (value.size != spaces.size) return false

        return spaces.all { (key, space) ->
            value.containsKey(key) && space.contains(value[key])
        }
    }
}
