package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class DPQFunctionPrediction<State, Action>(
    initialQ: QFunction<State, Action>,
    private val estimator: DPQFunctionEstimator<State, Action>,
    private val model: MDPModel<State, Action>,
    private val onQFunctionUpdate: (QFunction<State, Action>) -> Unit = {}
) {
    var q: QFunction<State, Action> = initialQ
        private set

    fun evaluate(policy: Policy<State, Action>): QFunction<State, Action> {
        val trajectory = model.allStates().flatMap { s ->
            model.transitions(s, policy(s))
        }

        q = estimator.estimate(q, trajectory)
        onQFunctionUpdate(q)
        return q
    }
}
