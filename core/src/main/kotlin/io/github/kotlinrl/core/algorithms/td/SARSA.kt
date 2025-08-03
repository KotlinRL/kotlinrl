package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class SARSA<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : TabularTDAlgorithm<State, Action>(initialPolicy, initialQ, alpha, gamma, onQFunctionUpdate, onPolicyUpdate) {

    val estimator = SARSAQFunctionEstimator<State, Action>(alpha, gamma)

    override fun observe(transition: Transition<State, Action>) {
        q = estimator.estimate(q, transition)
        policy = improvement(q)
    }
}



