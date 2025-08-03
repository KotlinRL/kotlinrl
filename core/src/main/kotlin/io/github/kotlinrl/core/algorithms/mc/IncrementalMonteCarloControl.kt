package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.StateActionKeyFunction
import io.github.kotlinrl.core.algorithms.defaultStateActionKeyFunction

class IncrementalMonteCarloControl<State, Action>(
    initialPolicy: Policy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule = ParameterSchedule { 0.05},
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultStateActionKeyFunction,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : MonteCarloAlgorithm<State, Action>(initialPolicy, initialQ, gamma, onQFunctionUpdate, onPolicyUpdate) {
    private val evaluator = IncrementalMonteCarloQFunctionEstimator(
        gamma = gamma,
        alpha = alpha,
        firstVisitOnly = firstVisitOnly,
        stateActionKeyFunction = stateActionKeyFunction
    )

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        q = evaluator.estimate(q, trajectory)
        policy = improvement(q)
    }
}
