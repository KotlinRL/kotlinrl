package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*
import kotlin.math.*

class NStepSARSAQFunctionEstimator<State, Action>(
    private val q: QFunction<State, Action>,
    private val alpha: Double,
    private val gamma: Double,
    private val policy: StochasticPolicy<State, Action>,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
) : NStepTDQFunctionEstimator<State, Action> {

    override fun estimate(trajectory: Trajectory<State, Action>): QFunction<State, Action> {
        val policyProbabilities = policy.asPolicyProbabilities(stateActionListProvider)

        val (s0, a0) = trajectory.first().state to trajectory.first().action

        var g = 0.0
        for (i in trajectory.indices) {
            g += gamma.pow(i) * trajectory[i].reward
        }

        val last = trajectory.last()
        if (!last.done) {
            val expectedQ = policyProbabilities(last.state).entries.sumOf { (a, prob) ->
                prob * q[last.state, a]
            }
            g += gamma.pow(trajectory.size) * expectedQ
        }

        val currentQ = q[s0, a0]
        val updated = currentQ + alpha * (g - currentQ)
        return q.update(s0, a0, updated)
    }
}
