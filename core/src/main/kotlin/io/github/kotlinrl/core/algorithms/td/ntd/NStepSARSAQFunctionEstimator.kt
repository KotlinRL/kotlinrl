package io.github.kotlinrl.core.algorithms.td.ntd

import io.github.kotlinrl.core.*
import kotlin.math.*

class NStepSARSAQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
) : TrajectoryQFunctionEstimator<State, Action> {

    var policy = initialPolicy

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {

        val (s0, a0) = trajectory.first().state to trajectory.first().action

        var g = 0.0
        for (i in trajectory.indices) {
            g += gamma.pow(i) * trajectory[i].reward
        }

        val last = trajectory.last()
        if (!last.done) {
            val expectedQ = policy.probabilities(last.state).entries.sumOf { (a, prob) ->
                prob * Q[last.state, a]
            }
            g += gamma.pow(trajectory.size) * expectedQ
        }

        val currentQ = Q[s0, a0]
        val updated = currentQ + alpha() * (g - currentQ)
        return Q.update(s0, a0, updated)
    }
}
