package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class DPValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val model: MDPModel<State, Action>,
    private val estimator: DPValueFunctionEstimator<State, Action>
) {

    var V: EnumerableValueFunction<State> = initialV
        private set

    operator fun invoke(policy: Policy<State, Action>): EnumerableValueFunction<State> {
        val transitions = model.allStates().flatMap { s ->
            model.transitions(s, policy(s))
        }

        V = estimator.estimate(V, transitions)
        return V
    }
}
