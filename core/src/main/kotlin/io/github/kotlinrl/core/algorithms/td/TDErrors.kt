package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

object TDErrors {
    fun <State, Action> qLearning(): TDError<State, Action> =
    TDError { Q, t, _, gamma, done ->
        val (s, a, r, sPrime) = t
        val nextQ = if (t.done) 0.0 else Q.maxValue(sPrime)
        r + gamma * nextQ - Q[s, a]
    }

    fun <State, Action> sarsa(): TDError<State, Action> =
        TDError { Q, t, aPrime, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q[sPrime, aPrime!!]
            r + gamma * nextQ - Q[s, a]
        }

    fun <State, Action> expectedSarsa(policy: QFunctionPolicy<State, Action>): TDError<State, Action> =
        TDError { _, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val Q = policy.Q
            val expectedQ = if (!t.done) {
                policy.probabilities(sPrime).entries.sumOf { (aPrime, prob) ->
                    prob * Q[sPrime, aPrime]
                }
            } else 0.0
            r + gamma * expectedQ - Q[s, a]
        }
}