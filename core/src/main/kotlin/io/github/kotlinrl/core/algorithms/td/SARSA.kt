package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class SARSA<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    alpha: ParameterSchedule,
    gamma: Double
) : TabularTDAlgorithm<State, Action>(initialPolicy, initialQ, improvement, onQFunctionUpdate, onPolicyUpdate, alpha, gamma) {

    val estimator = SARSAQFunctionEstimator<State, Action>(alpha, gamma)

    override fun observe(transition: Transition<State, Action>) {
        val updatedQ = estimator.estimate(q, transition)
        updatedQFunction(updatedQ)
        improvePolicy()
    }
}



