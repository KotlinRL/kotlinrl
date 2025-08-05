package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

object TDErrors {
    fun <State, Action> qLearning(): TDError<State, Action> =
    TDError { q, t, _, gamma, done ->
        val (s, a, r, sPrime) = t
        val nextQ = if (t.done) 0.0 else q.maxValue(sPrime)
        r + gamma * nextQ - q[s, a]
    }

    fun <State, Action> sarsa(): TDError<State, Action> =
        TDError { q, t, aPrime, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else q[sPrime, aPrime!!]
            r + gamma * nextQ - q[s, a]
        }

    fun <State, Action> expectedSarsa(policy: QFunctionPolicy<State, Action>): TDError<State, Action> =
        TDError { _, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val q = policy.q
            val expectedQ = if (!t.done) {
                policy.probabilities(sPrime).entries.sumOf { (aPrime, prob) ->
                    prob * q[sPrime, aPrime]
                }
            } else 0.0
            r + gamma * expectedQ - q[s, a]
        }
}