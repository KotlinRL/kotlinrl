package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.StateActionKeyFunction
import io.github.kotlinrl.core.algorithms.defaultStateActionKeyFunction

class IncrementalMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    gamma: Double = 0.99,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    alpha: ParameterSchedule = ParameterSchedule { 0.05},
    firstVisitOnly: Boolean = true
) : MonteCarloAlgorithm<State, Action>(initialPolicy, initialQ, improvement, gamma, onQFunctionUpdate, onPolicyUpdate) {
    private val evaluator = IncrementalMonteCarloQFunctionEstimator(
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly,
        stateActionKeyFunction = stateActionKeyFunction
    )

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        val currentQ = evaluator.estimate(q, trajectory, episode)
        updatedQFunction(currentQ)
        improvePolicy()
    }
}
