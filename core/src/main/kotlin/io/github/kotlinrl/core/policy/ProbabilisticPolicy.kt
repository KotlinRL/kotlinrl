package io.github.kotlinrl.core.policy

import kotlin.random.*

abstract class ProbabilisticPolicy<S, A>(
    private val rng: Random
) : Policy<S, A> {

    protected fun calculateAndSample(scores: List<Double>, actions: List<A>): A {
        val probabilities = computeProbabilities(scores)
        return sample(actions, probabilities)
    }

    private fun computeProbabilities(scores: List<Double>): List<Double> {
        val sum = scores.sum()
        return scores.map { it / sum }
    }

    private fun sample(actions: List<A>, probabilities: List<Double>): A {
        val rand = rng.nextDouble()
        var cumulative = 0.0
        for (i in actions.indices) {
            cumulative += probabilities[i]
            if (rand < cumulative) return actions[i]
        }
        return actions.last()
    }
}