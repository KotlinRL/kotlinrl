package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepTDValueFunctionEstimator<State, Action>(
    private val alpha: ParameterSchedule,
    private val gamma: Double,
    private val td: NStepTDVError<State> = NStepTDVErrors.nStep()
) : TrajectoryValueFunctionEstimator<State, Action> {

    override fun estimate(V: EnumerableValueFunction<State>, trajectory: Trajectory<State, Action>): EnumerableValueFunction<State> {
        if (trajectory.isEmpty()) return V

        val s0 = trajectory.first().state
        val delta = td(V, trajectory, gamma)
        val updated = V[s0] + alpha() * delta
        return V.update(s0, updated)
    }
}
