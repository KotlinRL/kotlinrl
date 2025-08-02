package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDZeroQFunctionEstimator<State, Action>(
    private val alpha: Double,
    private val gamma: Double,
    private val probabilities: PolicyProbabilities<State, Action>
) : TDQFunctionEstimator<State, Action> {
    override fun estimate(q: QFunction<State, Action>, transition: Transition<State, Action>): QFunction<State, Action> {
        val (s, a, r, sPrime) = transition
        val done = transition.done

        val current = q[s, a]
        val nextValue = if (done) 0.0 else {
            val actionProbs = probabilities(sPrime)
            actionProbs.entries.sumOf { (aPrime, prob) -> prob * q[sPrime, aPrime] }
        }

        val updated = current + alpha * (r + gamma * nextValue - current)
        return q.update(s, a, updated)
    }
}
