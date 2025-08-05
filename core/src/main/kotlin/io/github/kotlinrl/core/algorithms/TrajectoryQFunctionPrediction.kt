package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.EnumerableQFunction
import io.github.kotlinrl.core.EnumerableQFunctionUpdate
import io.github.kotlinrl.core.Trajectory
import io.github.kotlinrl.core.TrajectoryObserver
import io.github.kotlinrl.core.Transition
import io.github.kotlinrl.core.TransitionObserver

class TrajectoryQFunctionPrediction<State, Action>(
    initialQ: EnumerableQFunction<State, Action>,
    private val estimator: TrajectoryQFunctionEstimator<State, Action>,
) : TrajectoryObserver<State, Action> {

    var q = initialQ
        private set

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        q = estimator.estimate(q, trajectory)
    }
}