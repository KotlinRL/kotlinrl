package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepSARSA<State, Action>(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    private val n: Int,
    stateActionListProvider: StateActionListProvider<State, Action>,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
) : TabularTDAlgorithm<State, Action>(initialPolicy, initialQ, alpha, gamma, onQFunctionUpdate, onPolicyUpdate) {

    private val queue = ArrayDeque<Transition<State, Action>>()
    val estimator = NStepSARSAQFunctionEstimator(
        alpha = alpha,
        gamma = gamma,
        policyProbabilities = initialPolicy.asPolicyProbabilities(stateActionListProvider)
    )

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
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
        q = estimator.estimate(q, queue.take(steps))
        queue.removeFirst()
        policy = improvement(q)
        onQFunctionUpdate(q)
        onPolicyUpdate(policy)
    }
}
