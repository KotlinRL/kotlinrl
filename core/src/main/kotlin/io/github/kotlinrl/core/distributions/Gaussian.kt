package io.github.kotlinrl.core.distributions

import kotlin.math.*
import kotlin.random.*

/**
 * Represents a Gaussian (normal) probability distribution defined by a mean and standard deviation.
 * This distribution is continuous and has infinite support with a bell-shaped curve.
 *
 * @constructor Creates a Gaussian distribution with the specified mean and standard deviation.
 * Requires the standard deviation to be greater than zero.
 *
 * @param mean The mean (μ) of the Gaussian distribution. Determines the center of the distribution.
 * @param std The standard deviation (σ) of the Gaussian distribution. Determines the spread of the distribution.
 */
class Gaussian(val mean: Double, val std: Double) : Distribution<Double> {
    init {
        require(std > 0)
    }

    private val var2 = std * std

    /**
     * Computes the probability density function (PDF) of the Gaussian distribution for a given value.
     * The PDF represents the likelihood of a random variable taking the given value `t` under the
     * Gaussian distribution defined by its mean and standard deviation.
     *
     * @param t The value for which the probability density function is to be calculated.
     * @return The probability density of the Gaussian distribution at the given value `t`.
     */
    override fun prob(t: Double): Double =
        (1.0 / (std * sqrt(2 * Math.PI))) * exp(-0.5 * (t - mean) * (t - mean) / var2)

    /**
     * Returns the support set of the Gaussian distribution.
     *
     * The support of a Gaussian distribution is infinite, as the probability density
     * is non-zero for all real numbers, extending to negative and positive infinity.
     * In this implementation, an empty set is returned to signify infinite support.
     *
     * @return An empty set representing the infinite support of the Gaussian distribution.
     */
    override fun support(): Set<Double> = emptySet() // infinite support

    /**
     * Generates a random sample from a Gaussian (normal) distribution
     * based on the specified mean and standard deviation of the distribution.
     *
     * @param rng The random number generator used to generate the sample.
     * @return A random value sampled from the Gaussian distribution.
     */
    override fun sample(rng: Random) = rng.nextGaussian(mean, std)

    /**
     * Computes the natural logarithm of the probability density function (PDF) of the Gaussian distribution
     * for a given value.
     *
     * The logarithm of the probability density is useful in many scenarios, such as numerical stability
     * when working with very small probabilities or in optimization problems where log-probabilities are preferred.
     *
     * @param t The value for which the log-probability density function is to be calculated.
     * @return The natural logarithm of the probability density of the Gaussian distribution at the given value `t`.
     */
    override fun logProb(t: Double) =
        -ln(std) - 0.5 * ln(2 * Math.PI) - 0.5 * ((t - mean) * (t - mean) / var2)
}
