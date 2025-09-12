package io.github.kotlinrl.core.space

import kotlin.random.*

/**
 * Represents a space of text strings with a maximum length, defined character set, and optional random seed.
 * This class generates random text samples and checks for membership of values in the defined space.
 *
 * @property maxLength The maximum allowable length for strings in the space.
 * @property charset A set of characters that can appear in generated strings. Defaults to all printable ASCII characters.
 * @property seed An optional seed for the random number generator to enable deterministic behavior.
 */
class Text(
    val maxLength: Int,
    val charset: Set<Char> = (' '..'~').toSet(),
    val seed: Int? = null
) : Space<String> {

    override val random: Random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Generates a random string sample from the defined text space using the specified character set and maximum length.
     * The length of the generated string is randomly determined up to the maximum length.
     *
     * @return A randomly generated string consisting of characters from the character set, with a length between 0 and the maximum length (inclusive).
     */
    override fun sample(): String {
        val length = random.nextInt(maxLength + 1)
        return (1..length)
            .map { charset.random(random) }
            .joinToString("")
    }

    /**
     * Checks whether the given value is contained within the defined text space.
     * The value is considered to be contained if it is a string, its length does not exceed the maximum allowable length,
     * and all its characters are part of the defined character set.
     *
     * @param value The value to check. It should be `Any?`, but only strings are eligible for membership in the space.
     * @return `true` if the value is a string, its length is less than or equal to `maxLength`,
     *         and all characters are in the `charset`. Otherwise, `false`.
     */
    override fun contains(value: Any?): Boolean =
        value is String &&
                value.length <= maxLength &&
                value.all { it in charset }
}
