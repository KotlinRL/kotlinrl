package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class TDValueFunctionPrediction<State>(
    v: ValueFunction<State>,
    private val estimator: TDValueFunctionEstimator<State>,
    private val onValueFunctionUpdate: (ValueFunction<State>) -> Unit = { },
) : TransitionObserver<State, Any?> {
    var valueFunction = v
        private set

    override fun invoke(transition: Transition<State, Any?>) {
        valueFunction = estimator.estimate(valueFunction, transition)
        onValueFunctionUpdate(valueFunction)
    }
}
