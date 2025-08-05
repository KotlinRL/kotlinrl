package io.github.kotlinrl.core.algorithms.td.ntd

import io.github.kotlinrl.core.*

class NStepSARSA<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    private val n: Int,
    estimator: TrajectoryQFunctionEstimator<State, Action> = NStepSARSAQFunctionEstimator(
        initialPolicy = initialPolicy,
        alpha = alpha,
        gamma = gamma,
    ),
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = { },
    onPolicyUpdate: PolicyUpdate<State, Action> = { },
) : TrajectoryQFunctionAlgorithm<State, Action>(
    initialPolicy = initialPolicy,
    estimator = estimator,
    onQFunctionUpdate = onQFunctionUpdate,
    onPolicyUpdate = {
        when (estimator) {
            is NStepSARSAQFunctionEstimator -> estimator.policy = it as QFunctionPolicy<State, Action>
        }
        onPolicyUpdate(it)
    }
) {

    private val queue = ArrayDeque<Transition<State, Action>>()
    private var episode = 0

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        this.episode = episode
        while (queue.isNotEmpty()) {
            calculate(queue.size)
        }
    }

    override fun observe(transition: Transition<State, Action>) {
        queue.addLast(transition)
        if (queue.size >= n) {
            calculate(n)
        }
    }

    private fun calculate(steps: Int) {
        super.observe(queue.take(steps), episode)
        queue.removeFirst()
    }
}
