package io.github.kotlinrl.core.distributions

import kotlin.random.*

/**
 * Represents a Delta distribution, a discrete probability distribution where all probability
 * mass is concentrated on a single value `x`.
 *
 * @param T The type of the value `x` around which the delta distribution is defined.
 * @property x The value for which the probability is always 1.0, while it is 0.0 for all other values.
 */
class Delta<T>(val x: T) : Distribution<T> {
    /**
     * Computes the probability of the given value `t` in the context of the delta distribution.
     * In a delta distribution, all probability mass is concentrated on a single value `x`.
     * This method returns 1.0 if `t` equals `x`, and 0.0 otherwise.
     *
     * @param t The value for which the probability is to be computed.
     * @return The probability of `t`, which is 1.0 if `t` equals `x` and 0.0 otherwise.
     */
    override fun prob(t: T) = if (t == x) 1.0 else 0.0

    /**
     * Returns the support set of the delta distribution.
     *
     * In a delta distribution, the support is a set containing the single value `x`
     * where all the probability mass is concentrated.
     *
     * @return A set containing the single value `x`.
     */
    override fun support() = setOf(x)

    /**
     * Samples a value from the delta distribution.
     * Since all probability mass is concentrated on a single value `x`,
     * this method will always return `x` regardless of the provided random number generator.
     *
     * @param rng The random number generator that could provide randomness (unused in this case).
     * @return The value `x` on which the delta distribution is centered.
     */
    override fun sample(rng: Random) = x
}