package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepTDQFunctionEstimator<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>? = null,
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: NStepTDQError<State, Action>
) : TrajectoryQFunctionEstimator<State, Action> {

    var policy = initialPolicy
    var tailAction: Action? = null

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        if (trajectory.isEmpty()) return Q

        val s0 = trajectory.first().state
        val a0 = trajectory.first().action

        val delta = td(Q, trajectory, policy, tailAction, gamma)
        val updated = Q[s0, a0] + alpha() * delta
        return Q.update(s0, a0, updated)
    }
}
