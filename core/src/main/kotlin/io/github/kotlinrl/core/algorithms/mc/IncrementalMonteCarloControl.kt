package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class IncrementalMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule = ParameterSchedule { 0.05 },
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : MonteCarloAlgorithm<State, Action>(initialPolicy, initialQ, gamma, onQFunctionUpdate, onPolicyUpdate) {
    private val evaluator = IncrementalMonteCarloQFunctionEstimator<State, Action>(
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly
    )

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        q = evaluator.estimate(q, trajectory)
        policy = improvement(q)
        onQFunctionUpdate(q)
        onPolicyUpdate(policy)
    }
}
