package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class ExpectedSARSA<State, Action>(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : TabularTDAlgorithm<State, Action>(initialPolicy, initialQ, alpha, gamma, onQFunctionUpdate, onPolicyUpdate) {
    val estimator = ExpectedSARSAQFunctionEstimator(
        alpha = alpha,
        gamma = gamma,
        policyProbabilities = initialPolicy.asPolicyProbabilities(stateActionListProvider),
        stateActionListProvider = stateActionListProvider)

    override fun observe(transition: Transition<State, Action>) {
        q = estimator.estimate(q, transition)
        policy = improvement(q)
    }
}
