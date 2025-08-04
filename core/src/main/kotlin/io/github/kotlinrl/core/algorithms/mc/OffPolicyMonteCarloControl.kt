package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.Policy
import io.github.kotlinrl.core.QFunction
import io.github.kotlinrl.core.StochasticPolicy
import io.github.kotlinrl.core.policy.PolicyImprovementStrategy

class OffPolicyMonteCarloControl<State, Action>(
    initialQ: QFunction<State, Action>,
    behavioralPolicy: StochasticPolicy<State, Action>,
    targetPolicy: Policy<State, Action>,
    gamma: Double,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : MonteCarloAlgorithm<State, Action>(behavioralPolicy, initialQ, gamma, onQFunctionUpdate, onPolicyUpdate) {
    @Suppress("UNCHECKED_CAST")
    private val targetImprovement = targetPolicy as PolicyImprovementStrategy<State, Action>

    val evaluator = OffPolicyMonteCarloQFunctionEstimator(
        initTargetPolicy = targetPolicy,
        behaviorPolicy = behavioralPolicy,
        gamma = gamma,
    )

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        q = evaluator.estimate(q, trajectory)
        policy = improvement(q)
        evaluator.targetPolicy = targetImprovement(q)
        onQFunctionUpdate(q)
        onPolicyUpdate(policy)
        onPolicyUpdate(evaluator.targetPolicy)
    }
}