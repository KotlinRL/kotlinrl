package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class DPQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: DPQFunctionEstimator<State, Action>,
    private val model: MDPModel<State, Action>,
    private val onQFunctionUpdate: (EnumerableQFunction<State, Action>) -> Unit = {}
) {
    var Q: EnumerableQFunction<State, Action> = initialQ
        private set(value) {
            field = value
            onQFunctionUpdate(value)
        }

    fun evaluate(policy: Policy<State, Action>): EnumerableQFunction<State, Action> {
        val trajectory = model.allStates().flatMap { s ->
            model.transitions(s, policy(s))
        }

        Q = estimator.estimate(Q, trajectory)
        return Q
    }
}
