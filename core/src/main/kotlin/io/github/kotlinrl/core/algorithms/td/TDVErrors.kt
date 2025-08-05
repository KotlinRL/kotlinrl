package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

object TDVErrors {
    // On-policy TD(0) for V
    fun <State> tdZero(): TDVError<State> =
        TDVError { V, t, gamma ->
            val (s, _, r, sPrime) = t
            val boot = if (t.done) 0.0 else V[sPrime]
            r + gamma * boot - V[s]
        }

    // Off-policy TD(0) for V with per-step importance weight rho
    fun <State> tdZeroWeighted(
        rho: (Transition<State, *>) -> Double = { 1.0 }
    ): TDVError<State> =
        TDVError { V, t, gamma ->
            val (s, _, r, sPrime) = t
            val boot = if (t.done) 0.0 else V[sPrime]
            rho(t) * (r + gamma * boot - V[s])
        }
}