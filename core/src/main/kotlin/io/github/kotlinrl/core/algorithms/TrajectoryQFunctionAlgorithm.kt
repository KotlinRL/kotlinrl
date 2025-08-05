package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.*

abstract class TrajectoryQFunctionAlgorithm<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    estimator: TrajectoryQFunctionEstimator<State, Action>,
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { }
) : QFunctionAlgorithm<State, Action>(initialPolicy, onPolicyUpdate, onQFunctionUpdate) {

    protected val prediction = TrajectoryQFunctionPrediction(q, estimator)

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        prediction(trajectory, episode)
        q = prediction.q
        policy = policy.improve(q)
    }
}