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
    private val cum: DoubleArray

    init {
        require(outcomes.isNotEmpty())
        require(outcomes.size == probs.size)
        require(probs.all { it >= 0.0 })

        val s = probs.sum()
        require(s > 0.0) { "All probabilities are zero." }
        val norm = probs / s
        val tmp = DoubleArray(norm.size)
        var acc = 0.0
        for (i in 0 until norm.size) {
            acc += norm[i]
            tmp[i] = acc
        }
        tmp[tmp.lastIndex] = 1.0
        cum = tmp
    }

    override fun prob(t: T): Double {
        val idx = outcomes.indexOf(t)
        if (idx < 0) return 0.0
        val prev = if (idx == 0) 0.0 else cum[idx - 1]
        return cum[idx] - prev
    }

    override fun support(): Set<T> =
        outcomes.filterIndexed { i, _ -> (if (i == 0) cum[i] else cum[i] - cum[i - 1]) > 0.0 }.toSet()

    override fun sample(rng: Random): T {
        val r = rng.nextDouble() // in [0,1)
        val i = cum.binarySearch(r).let { if (it < 0) -(it + 1) else it }
        return outcomes[i]
    }
}
