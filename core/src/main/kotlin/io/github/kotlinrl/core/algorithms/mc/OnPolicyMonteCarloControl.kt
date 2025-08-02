package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.defaultStateActionKeyFunction

class OnPolicyMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    gamma: Double,
    firstVisitOnly: Boolean = true,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { }
) : MonteCarloAlgorithm<State, Action>(initialPolicy, initialQ, improvement, gamma, onQFunctionUpdate, onPolicyUpdate) {
    private val evaluator = OnPolicyMonteCarloQFunctionEstimator(gamma, stateActionKeyFunction, firstVisitOnly)

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        val currentQ = evaluator.estimate(q, trajectory)
        updatedQFunction(currentQ)
        improvePolicy()
    }
}

