package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OnPolicyMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
) : MonteCarloAlgorithm<State, Action>(initialPolicy, initialQ, gamma, onQFunctionUpdate, onPolicyUpdate) {
    private val evaluator = OnPolicyMonteCarloQFunctionEstimator<State, Action>(gamma, firstVisitOnly)

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        q = evaluator.estimate(q, trajectory)
        policy = improvement(q)
        onQFunctionUpdate(q)
        onPolicyUpdate(policy)
    }
}

