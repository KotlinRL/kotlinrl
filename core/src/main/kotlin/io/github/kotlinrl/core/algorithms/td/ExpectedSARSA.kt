package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.policy.StateActionListProvider

class ExpectedSARSA<State, Action>(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    alpha: ParameterSchedule,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>
) : TabularTDAlgorithm<State, Action>(initialPolicy, initialQ, improvement, onQFunctionUpdate, onPolicyUpdate, alpha, gamma) {
    val estimator = ExpectedSARSAQFunctionEstimator(
        alpha = alpha(),
        gamma = gamma,
        initialPolicyProbabilities = (policy as StochasticPolicy<State, Action>).asPolicyProbabilities(stateActionListProvider),
        stateActionListProvider = stateActionListProvider)

    override fun observe(transition: Transition<State, Action>) {
        val updatedQ = estimator.estimate(q, transition)
        updatedQFunction(updatedQ)
        improvePolicy()
        val policyProbabilities = (policy as StochasticPolicy<State, Action>).asPolicyProbabilities(stateActionListProvider)
        estimator.policyProbabilities = policyProbabilities
    }
}
