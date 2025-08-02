package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.StateActionKeyFunction
import io.github.kotlinrl.core.algorithms.defaultStateActionKeyFunction

class OffPolicyMonteCarloControl<State, Action>(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    targetPolicy: Policy<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    gamma: Double,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : MonteCarloAlgorithm<State, Action>(initialPolicy, initialQ, improvement, gamma, onQFunctionUpdate, onPolicyUpdate) {
    private var currentQ = initialQ

    val evaluator = OffPolicyMonteCarloQFunctionEstimator(
        initTargetPolicy = targetPolicy,
        behaviorPolicy = policy as StochasticPolicy<State, Action>,
        gamma = gamma,
        stateActionKeyFunction = stateActionKeyFunction
    )

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        currentQ = evaluator.estimate(currentQ, trajectory)
        val currentTargetPolicy = improvement(currentQ)
        onQFunctionUpdate(currentQ)
        onPolicyUpdate(currentTargetPolicy)
        evaluator.targetPolicy = currentTargetPolicy
    }
}