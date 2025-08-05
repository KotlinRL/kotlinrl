package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * Represents a structure for storing a single value associated with an index.
 *
 * This class is often used to define a sampled result or data point
 * where an indexed identifier is linked to a value. The index typically serves
 * as a means of discrimination or categorization of the value.
 *
 * @property index The index associated with the value. Typically used to categorize or identify the value.
 * @property value The actual data or element associated with the index. Can be of any type.
 */
data class OneOfSample(
    val index: Int,
    val value: Any
)

/**
 * A space that represents a collection of subspaces, where samples are drawn randomly
 * from one of the subspaces based on a uniform probability distribution.
 *
 * The `OneOf` class models a space made up of multiple subspaces, enabling the generation of
 * samples by randomly selecting one of these subspaces and sampling from it. This can be used
 * to define composite spaces where the sampling logic depends on subspaces with potentially
 * different structures and types.
 *
 * @property spaces The list of subspaces that make up the composite space. Each subspace must
 *                  implement the `Space` interface.
 * @property seed An optional seed value used to initialize the random number generator
 *                for deterministic sampling. If null, a default random generator is used.
 */
class OneOf(
    val spaces: List<Space<*>>,
    val seed: Int? = null
) : Space<OneOfSample> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Samples a value from one of the subspaces within the composite space. The subspace
     * is selected randomly based on a uniform distribution, and a sample is drawn from it.
     *
     * @return A `OneOfSample` instance containing the index of the selected subspace and
     * the sampled value from that subspace.
     */
    override fun sample(): OneOfSample {
        val which = random.nextInt(spaces.size)
        val value = spaces[which].sample()!!
        return OneOfSample(which, value)
    }

    /**
     * Checks whether the provided value is contained within the composite space.
     * The value is considered contained if it is of type `OneOfSample`, its index
     * is within the range of available subspaces, and the associated subspace
     * contains the value.
     *
     * @param value The value to check for membership in the composite space.
     *              It must be of type `OneOfSample` with a valid index and value.
     * @return `true` if the value is contained in the composite space, `false` otherwise.
     */
    override fun contains(value: Any?): Boolean =
        value is OneOfSample &&
                value.index in spaces.indices &&
                spaces[value.index].contains(value.value)
}
