package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

abstract class NStepTabularTDAlgorithm<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    protected val alpha: ParameterSchedule,
    protected val gamma: Double,
    protected val estimator: NStepTDQFunctionEstimator<State, Action>
) : QFunctionAlgorithm<State, Action>(initialPolicy, initialQ, onPolicyUpdate, onQFunctionUpdate) {

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        q = estimator.estimate(q, trajectory)
        policy = improvement(q)
    }
}