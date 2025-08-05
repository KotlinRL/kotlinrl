package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

object TDQErrors {
    fun <State, Action> qLearning(): TDQError<State, Action> =
        TDQError { Q, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q.maxValue(sPrime)
            r + gamma * nextQ - Q[s, a]
        }

    fun <State, Action> sarsa(): TDQError<State, Action> =
        TDQError { Q, t, aPrime, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q[sPrime, aPrime!!]
            r + gamma * nextQ - Q[s, a]
        }

    fun <State, Action> expectedSarsa(policy: QFunctionPolicy<State, Action>): TDQError<State, Action> =
        TDQError { _, t, _, gamma, done ->
            val (s, a, r, sPrime) = t
            val Q = policy.Q
            val expectedQ = if (!t.done) {
                policy.probabilities(sPrime).entries.sumOf { (aPrime, prob) ->
                    prob * Q[sPrime, aPrime]
                }
            } else 0.0
            r + gamma * expectedQ - Q[s, a]
        }

    fun <State, Action> semiGradientSarsa(): TDQError<State, Action> =
        TDQError { Q, t, aPrime, gamma, done ->
            val (s, a, r, sPrime) = t
            val nextQ = if (t.done) 0.0 else Q[sPrime, aPrime!!]
            r + gamma * nextQ - Q[s, a]
        }
}