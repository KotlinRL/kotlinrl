package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.TransitionQFunctionEstimator

class TDZeroQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val alpha: Double,
    private val gamma: Double
) : TransitionQFunctionEstimator<State, Action> {

    val policy = initialPolicy

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        transition: Transition<State, Action>
    ): EnumerableQFunction<State, Action> {
        val (s, a, r, sPrime) = transition
        val done = transition.done

        val current = Q[s, a]
        val nextValue = if (done) 0.0 else {
            val actionProbs = policy.probabilities(sPrime)
            actionProbs.entries.sumOf { (aPrime, prob) -> prob * Q[sPrime, aPrime] }
        }

        val updated = current + alpha * (r + gamma * nextValue - current)
        return Q.update(s, a, updated)
    }
}
