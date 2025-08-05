package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class DPValueFunctionPrediction<State, Action>(
    initialV: EnumerableValueFunction<State>,
    private val model: MDPModel<State, Action>,
    private val estimator: DPValueFunctionEstimator<State, Action>,
    private val onValueFunctionUpdate: EnumerableValueFunctionUpdate<State> = { },
) {

    var valueFunction: EnumerableValueFunction<State> = initialV
        private set(value) {
            field = value
            onValueFunctionUpdate(value)
        }

    fun evaluate(policy: Policy<State, Action>): EnumerableValueFunction<State> {
        val transitions = model.allStates().flatMap { s ->
            model.transitions(s, policy(s))
        }

        valueFunction = estimator.estimate(valueFunction, transitions)
        return valueFunction
    }
}
