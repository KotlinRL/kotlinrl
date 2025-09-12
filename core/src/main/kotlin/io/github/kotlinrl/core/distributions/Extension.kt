package io.github.kotlinrl.core.distributions

import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import kotlin.math.*
import kotlin.random.*

/**
 * Generates a random value from a Gaussian (normal) distribution with a specified mean and standard deviation.
 * This implementation uses the Box-Muller transform to generate the value.
 *
 * @param mean The mean (μ) of the Gaussian distribution.
 * @param std The standard deviation (σ) of the Gaussian distribution.
 * @return A random value sampled from the Gaussian distribution defined by the specified mean and standard deviation.
 */
fun Random.nextGaussian(mean: Double, std: Double): Double {
    // Box–Muller
    val u1 = nextDouble().coerceIn(1e-12, 1.0)
    val u2 = nextDouble()
    val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * Math.PI * u2)
    return mean + std * z0
}

/**
 * Creates a delta distribution where all probability mass is concentrated on a single value `x`.
 * The delta distribution has a probability of 1.0 for `x` and 0.0 for all other values.
 *
 * @param x The value around which the delta distribution is defined.
 * @return A delta distribution with all probability mass assigned to `x`.
 */
fun <T> Distribution.Companion.delta(x: T) = Delta(x)

/**
 * Constructs a uniform categorical distribution where each outcome is equally likely.
 *
 * @param outcomes A list of outcomes that form the support of the distribution. Each outcome
 * is assigned an equal probability such that the sum of all probabilities is 1.0.
 * @return A `Categorical` distribution with uniform probabilities assigned to each outcome.
 */
fun <T> Distribution.Companion.uniform(outcomes: List<T>) =
    Categorical(outcomes, mk.d1array<Double>(outcomes.size) { 1.0 / outcomes.size })

/**
 * Creates a Gaussian (normal) distribution with the specified mean and standard deviation.
 *
 * @param mean The mean (average) of the Gaussian distribution.
 * @param std The standard deviation of the Gaussian distribution, which must be greater than zero.
 * @return A `Gaussian` distribution with the specified parameters.
 */
fun Distribution.Companion.normal(mean: Double, std: Double) = Gaussian(mean, std)

fun <T> Distribution.Companion.categorical(outcomes: List<T>, probs: D1Array<Double>) = Categorical(outcomes, probs)
/**
 * Creates a categorical distribution from a list of outcomes, their corresponding logits,
 * and a boolean mask that determines which outcomes are considered valid.
 *
 * @param T The type of elements in the categorical distribution.
 * @param outcomes A list of outcomes for the categorical distribution.
 * @param logits An array of logits corresponding to the outcomes. Each logit represents the
 * relative unnormalized log-probability for its respective outcome.
 * @param mask A boolean array indicating which outcomes are valid and should contribute
 * to the resulting probability distribution. An outcome is included if its corresponding
 * mask value is true.
 * @return A `Categorical` distribution containing the valid outcomes and their normalized
 * probabilities derived from the logits and mask.
 */
fun <T> Distribution.Companion.maskedCategorical(outcomes: List<T>, logits: DoubleArray, mask: BooleanArray): Categorical<T> {
    val exps = mk.d1array<Double>(logits.size) { i -> if (mask[i]) exp(logits[i]) else 0.0 }
    val Z = exps.sum().takeIf { it > 0 } ?: 1.0
    val probs = mk.d1array<Double>(exps.size) { i -> exps[i] / Z }
    return Categorical(outcomes, probs)
}

/**
 * Constructs a `Categorical` distribution from a list of pairs, where each pair consists of an outcome
 * and its corresponding weight (non-negative value). Outcomes with a weight of zero or less are excluded
 * from the resulting distribution. Weights are normalized to ensure that probabilities sum to 1.
 *
 * @param S The type of the outcomes.
 * @param pairs A list of pairs where each pair consists of an outcome and its associated weight.
 *              The outcome is of type `S` and the weight is a `Double`. Only pairs with positive
 *              weights are considered.
 * @return A `Categorical` distribution constructed from the normalized weights of the provided pairs.
 */
fun <S> Distribution.Companion.categoricalFromPairs(pairs: List<Pair<S, Double>>): Categorical<S> {
    val filtered = pairs.filter { it.second > 0.0 }
    val norm = filtered.sumOf { it.second }.takeIf { it > 0 } ?: 1.0
    val outcomes = filtered.map { it.first }
    val probs = filtered.map { it.second / norm }.toDoubleArray()
    return Categorical(outcomes, mk.ndarray(probs))
}

/**
 * Constructs a categorical distribution from a map where each entry represents an outcome and its corresponding weight.
 * The weights are normalized to ensure the probabilities sum to 1, and entries with a weight of zero or less are excluded.
 *
 * @param S The type of the outcomes in the map.
 * @param mp A map where each key is an outcome (of type `S`) and each value is its associated weight (of type `Double`).
 *           Only entries with positive weights are considered.
 * @return A `Categorical` distribution created from the normalized weights of the provided map.
 */
fun <S> Distribution.Companion.categoricalFromMap(mp: Map<S, Double>) =
    categoricalFromPairs(mp.entries.map { it.key to it.value })

/**
 * Creates a diagonal Gaussian (normal) distribution with the specified mean and standard deviation vectors.
 * The diagonal Gaussian assumes independence between its dimensions, where each dimension is represented
 * by a univariate Gaussian distribution with its own mean and standard deviation.
 *
 * @param mean The mean vector of the Gaussian distribution. Each element corresponds to the mean
 * of a specific dimension.
 * @param std The standard deviation vector of the Gaussian distribution. Each element represents
 * the standard deviation of the respective dimension. All values in this array must be positive.
 * @throws IllegalArgumentException If the sizes of the mean and standard deviation arrays are not equal,
 * or if any standard deviation value is non-positive.
 * @return A `DiagGaussian` instance representing the diagonal Gaussian distribution with the specified parameters.
 */
fun Distribution.Companion.diagGaussian(mean: DoubleArray, std: DoubleArray) = DiagGaussian(mean, std)

/**
 * Creates a mixture distribution from a list of component distributions and their associated weights.
 * The weights determine the relative contribution of each component distribution to the overall mixture.
 *
 * @param components A list of component distributions that make up the mixture. Each component
 * distribution contributes to the overall probability distribution of the mixture.
 * @param weights An array of weights corresponding to the component distributions. The weights
 * should be non-negative and sum to 1.
 * @return A `Mixture` distribution composed of the given components and weights.
 */
fun Distribution.Companion.mixture(components: List<Distribution<Double>>, weights: D1Array<Double>) =
    Mixture(components, weights)