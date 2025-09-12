package io.github.kotlinrl.core.distributions

import kotlin.math.*
import kotlin.random.*

/**
 * Represents a probability distribution and extends the `Probabilities` interface.
 * A `Distribution` provides the capability to sample elements based on their probabilities
 * and to compute the log-probability of a specific element.
 *
 * @param T The type of elements that the distribution operates on.
 */
interface Distribution<T> {
    companion object { }
    /**
     * Computes the probability of a given element `a`.
     *
     * @param t The element for which the probability is calculated.
     * @return The probability of the element `a` as a value between 0.0 and 1.0.
     */
    fun prob(t: T): Double

    /**
     * Retrieves the probability of a given element `t` using the underlying probability calculation method.
     *
     * @param t The element for which the probability is to be retrieved.
     * @return The probability of the given element `t` as a value between 0.0 and 1.0.
     */
    operator fun get(t: T): Double = prob(t)

    /**
     * Retrieves the support set, which is the set of all elements with non-zero probability in
     * the distribution or probability model.
     *
     * @return A set of elements of type `T` that define the support of the distribution or probability model.
     */
    fun support(): Set<T>

    /**
     * Generates a sample from the probability distribution using the provided random number generator.
     *
     * @param rng A random number generator used to perform the sampling.
     * @return A sampled element of type A from the distribution.
     */
    fun sample(rng: Random): T

    /**
     * Computes the natural logarithm of the probability of a given element `a`.
     *
     * @param t The element for which the log-probability will be computed.
     * @return The natural logarithm of the probability of the element `a`.
     */
    fun logProb(t: T): Double = ln(prob(t))

}