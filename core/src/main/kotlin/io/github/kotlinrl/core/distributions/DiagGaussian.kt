package io.github.kotlinrl.core.distributions

import kotlin.math.*
import kotlin.random.*

/**
 * A multivariate diagonal Gaussian (normal) distribution.
 * This distribution represents a collection of independent Gaussian distributions, each defined
 * for a single dimension with provided mean and standard deviation values.
 *
 * @property mean The mean vector of the Gaussian distribution. Each element corresponds to the mean
 * of its respective dimension.
 * @property std The standard deviation vector of the Gaussian distribution. Each element corresponds
 * to the standard deviation of its respective dimension.
 *
 * @constructor Creates a diagonal Gaussian distribution given the mean and standard deviation vectors.
 * Ensures the mean and standard deviation vectors are of the same size, and all standard deviation values are positive.
 *
 * @throws IllegalArgumentException If the sizes of mean and standard deviation vectors are not equal,
 * or if any standard deviation value is non-positive.
 */
class DiagGaussian(
    val mean: DoubleArray,
    val std: DoubleArray
) : Distribution<DoubleArray> {
    init {
        require(mean.size == std.size); require(std.all { it > 0 })
    }

    /**
     * Computes the probability density of the given point `t` under the diagonal Gaussian distribution.
     * This method calculates the probability by exponentiating the log probability of `t`.
     *
     * @param t A multidimensional point represented as a `DoubleArray` for which the probability density
     * is to be computed. Each element in `t` corresponds to a value in the respective dimension of the
     * Gaussian distribution.
     * @return The probability density of the provided point `t` as a `Double`.
     */
    override fun prob(t: DoubleArray): Double = exp(logProb(t))

    /**
     * Computes the log probability of the given multidimensional point `t` under the diagonal Gaussian distribution.
     * The log probability is calculated using the mean and standard deviation for each dimension.
     *
     * @param t A multidimensional point represented as a `DoubleArray` for which the log probability is to be computed.
     * Each element in `t` corresponds to a value in the respective dimension of the Gaussian distribution.
     * @return The log probability of the provided point `t` as a `Double`.
     */
    override fun logProb(t: DoubleArray): Double {
        var lp = 0.0
        for (i in t.indices) {
            val z = (t[i] - mean[i]) / std[i]
            lp += -ln(std[i]) - 0.5 * ln(2 * Math.PI) - 0.5 * z * z
        }
        return lp
    }

    /**
     * Returns the support of the diagonal Gaussian distribution.
     *
     * The support of a diagonal Gaussian distribution is technically infinite,
     * but this method returns an empty set to indicate that the support is not
     * a discrete set of values.
     *
     * @return An empty set representing the support of the diagonal Gaussian distribution.
     */
    override fun support(): Set<DoubleArray> = emptySet()

    /**
     * Generates a random sample from a diagonal Gaussian (normal) distribution.
     * Each dimension of the output sample is independently drawn from a Gaussian distribution
     * defined by the corresponding mean and standard deviation.
     *
     * @param rng The random number generator used to produce the sample.
     * @return A multidimensional sample represented as a `DoubleArray` where each value is drawn
     * from a Gaussian distribution with the respective mean and standard deviation.
     */
    override fun sample(rng: Random): DoubleArray =
        DoubleArray(mean.size) { i -> rng.nextGaussian(mean[i], std[i]) }
}
