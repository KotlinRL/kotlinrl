package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*
import kotlin.math.*

class NStepSARSAValueFunctionEstimator<State, Action>(
    private val alpha: Double,
    private val gamma: Double,
) : NStepTDValueFunctionEstimator<State, Action> {

    override fun estimate(V: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        if (trajectory.isEmpty()) return V

        val s0 = trajectory.first().state
        val isTerminal = trajectory.last().done

        var g = 0.0
        for ((i, t) in trajectory.withIndex()) {
            g += gamma.pow(i) * t.reward
        }

        if (!isTerminal) {
            val sPrime = trajectory.last().nextState
            g += gamma.pow(trajectory.size) * V[sPrime]
        }

        val current = V[s0]
        val updated = current + alpha * (g - current)
        return V.update(s0, updated)
    }
}
