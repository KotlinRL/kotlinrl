package io.github.kotlinrl.core.distributions

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.random.*

/**
 * Represents a mixture distribution, where multiple component distributions are combined
 * using specified weights to form a new distribution.
 *
 * @param T The type of elements that the mixture distribution operates on.
 * @property components A list of component distributions that form the mixture.
 * @property weights An array of weights corresponding to each component distribution.
 * The weights should be non-negative and sum to 1.
 */
class Mixture<T>(
    val components: List<Distribution<T>>,
    val weights: D1Array<Double>
) : Distribution<T> {
    private val cat = Categorical((components.indices).toList(), weights.copy())

    /**
     * Computes the probability of a given element `t` within the mixture distribution.
     * The probability is calculated as the weighted sum of probabilities over all
     * component distributions in the mixture.
     *
     * @param t The element for which the probability is calculated.
     * @return The probability of the element `t` as a value between 0.0 and 1.0.
     */
    override fun prob(t: T): Double =
        components.indices.sumOf { k -> weights[k] * components[k].prob(t) }

    /**
     * Retrieves the combined support set of the mixture distribution.
     *
     * The support set of a mixture distribution is the union of the support sets of its component distributions.
     * It includes all elements that have non-zero probability in at least one of the component distributions.
     *
     * @return A set of elements of type `T` that represents the combined support of the mixture distribution.
     */
    override fun support(): Set<T> = components.flatMap { it.support() }.toSet()

    /**
     * Samples a random outcome from the mixture distribution using the provided random number generator.
     * The sampling is performed by first selecting a component distribution based on the categorical
     * distribution defined by the weights, and then sampling from the selected component distribution.
     *
     * @param rng A random number generator used to generate the random outcome.
     * @return A sampled element of type `T` from the mixture distribution.
     */
    override fun sample(rng: Random): T = components[cat.sample(rng)].sample(rng)
}
