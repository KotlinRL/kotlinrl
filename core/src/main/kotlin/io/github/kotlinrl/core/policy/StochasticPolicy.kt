package io.github.kotlinrl.core.policy

import kotlin.random.*

abstract class StochasticPolicy<State, Action>(
    private val rng: Random
) : ProbabilityFunction<State, Action>, Policy<State, Action> {

    abstract fun actionScores(state: State): List<Pair<Action, Double>>

    override fun invoke(state: State): Action {
        val (actions, scores) = actionScores(state).unzip()
        return calculateAndSample(scores, actions)
    }

    override fun invoke(state: State, action: Action): Double {
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

    private fun calculateAndSample(scores: List<Double>, actions: List<Action>): Action {
        val probs = computeProbabilities(scores)
        return sample(actions, probs)
    }

    private fun sample(actions: List<Action>, probabilities: List<Double>): Action {
        val rand = rng.nextDouble()
        var cumulative = 0.0
        for (i in actions.indices) {
            cumulative += probabilities[i]
            if (rand < cumulative) return actions[i]
        }
        return actions.last()
    }
}