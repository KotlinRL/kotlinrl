package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*
import kotlin.math.pow

object NStepTDVErrors {
    fun <State> nStep(): NStepTDVError<State> =
        NStepTDVError { V, traj, gamma ->
            if (traj.isEmpty()) return@NStepTDVError 0.0
            val s0 = traj.first().state
            val terminal = traj.last().done

            var G = 0.0
            for ((i, t) in traj.withIndex()) {
                G += gamma.pow(i) * t.reward
            }
            if (!terminal) {
                val sT = traj.last().nextState
                G += gamma.pow(traj.size) * V[sT]
            }
            G - V[s0]
        }

    // Optional: weighted/off-policy variant
    fun <State> nStepWeighted(
        rho: (Trajectory<State, *>) -> Double
    ): NStepTDVError<State> =
        NStepTDVError { V, traj, gamma ->
            rho(traj) * nStep<State>()(V, traj, gamma)
        }
}