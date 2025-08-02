package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class MonteCarloValueFunctionPrediction<State, Action>(
    initialV: ValueFunction<State>,
    private val estimator: MonteCarloValueFunctionEstimator<State, Action>,
    private val onValueFunctionUpdate: (ValueFunction<State>) -> Unit = { },
) : TrajectoryObserver<State, Action> {

    var valueFunction = initialV
        private set

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        valueFunction = estimator.estimate(valueFunction, trajectory)
        onValueFunctionUpdate(valueFunction)
    }
}
