package io.github.kotlinrl.core.policy

import kotlin.random.*

abstract class ProbabilisticPolicy<S, A>(
    private val rng: Random
) : StochasticPolicy<S, A> {

    abstract fun actionScores(state: S): List<Pair<A, Double>>

    override fun invoke(state: S): A {
        val (actions, scores) = actionScores(state).unzip()
        return calculateAndSample(scores, actions)
    }

    override fun probability(state: S, action: A): Double {
        val actionScoreList = actionScores(state)
        val actions = actionScoreList.map { it.first }
        val scores = actionScoreList.map { it.second }
        val probs = computeProbabilities(scores)
        return actions.indexOf(action).let { if (it >= 0) probs[it] else 0.0 }
    }

    protected fun computeProbabilities(scores: List<Double>): List<Double> {
        val sum = scores.sum()
        return scores.map { it / sum }
    }

    private fun calculateAndSample(scores: List<Double>, actions: List<A>): A {
        val probs = computeProbabilities(scores)
        return sample(actions, probs)
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