package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class MonteCarloQFunctionPrediction<State, Action>(
    initialQ: QFunction<State, Action>,
    private val estimator: MonteCarloQFunctionEstimator<State, Action>,
    private val onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { }
) : TrajectoryObserver<State, Action> {

    var q = initialQ
        private set

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        q = estimator.estimate(q, trajectory)
        onQFunctionUpdate(q)
    }
}
