package io.github.kotlinrl.core.distributions

import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*
import kotlin.random.*

/**
 * Represents a categorical distribution where each element in a finite set has an associated probability.
 *
 * @param T The type of elements in the categorical distribution.
 * @property outcomes A list of all possible outcomes in the distribution.
 *                    Each element in this list corresponds to a probabilistic outcome.
 * @property probs A one-dimensional array of probabilities corresponding to the outcomes.
 *                 The probabilities must be non-negative, sum to 1, and should have the same size as the outcomes list.
 * @throws IllegalArgumentException if the provided outcomes list is empty, if its size does not match the probs array,
 *                                  if the probabilities are negative, or if the probabilities do not sum to 1.
 */
class Categorical<T>(
    private val outcomes: List<T>,
    private val probs: D1Array<Double>
) : Distribution<T> {
    init {
        require(outcomes.isNotEmpty())
        require(outcomes.size == probs.size)
        val s = probs.sum();
        require(abs(s - 1.0) < 1e-9) { "Probs sum=$s" }
        require(probs.all { it >= 0.0 })
    }

    private val cum = probs.toDoubleArray()

    /**
     * Computes the probability of the specified outcome in the distribution.
     *
     * @param t The outcome for which the probability is to be computed.
     * @return The probability of the specified outcome. Returns 0.0 if the outcome is not present in the distribution.
     */
    override fun prob(t: T): Double {
        val idx = outcomes.indexOf(t);
        return if (idx >= 0) probs[idx] else 0.0
    }

    /**
     * Returns the set of all distinct outcomes present in the distribution.
     *
     * @return A set containing all unique outcomes supported by the distribution.
     */
    override fun support(): Set<T> = outcomes.filterIndexed { i, _ -> probs[i] > 0.0 }.toSet()

    /**
     * Samples a random outcome from the categorical distribution based on the provided random number generator.
     * The method uses the cumulative probability distribution to select an outcome corresponding to the sampled value.
     *
     * @param rng A random number generator used to sample the outcome.
     * @return A randomly sampled outcome of type `T` from the categorical distribution.
     */
    override fun sample(rng: Random): T {
        val r = rng.nextDouble()
        val i = cum.binarySearch(r).let { if (it < 0) -(it + 1) else it }
        return outcomes[min(i, outcomes.lastIndex)]
    }
}
