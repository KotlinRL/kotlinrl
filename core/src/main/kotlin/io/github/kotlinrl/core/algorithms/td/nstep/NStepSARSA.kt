package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

class NStepSARSA<State, Action>(
    initialPolicy: StochasticPolicy<State, Action>,
    initialQ: QFunction<State, Action>,
    improvement: PolicyImprovementStrategy<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double,
    onQFunctionUpdate: (QFunction<State, Action>) -> Unit = { },
    onPolicyUpdate: (Policy<State, Action>) -> Unit = { },
    private val n: Int,
    private val stateActionListProvider: StateActionListProvider<State, Action>
) : TabularTDAlgorithm<State, Action>(initialPolicy, initialQ, improvement, onQFunctionUpdate, onPolicyUpdate, alpha, gamma) {

    private val queue = ArrayDeque<Transition<State, Action>>()

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
        val estimator = NStepSARSAQFunctionEstimator(
            q = q,
            alpha = alpha(),
            gamma = gamma,
            policy = policy as StochasticPolicy<State, Action>,
            stateActionListProvider = stateActionListProvider
        )

        val updatedQ = estimator.estimate(queue.take(steps))
        queue.removeFirst()
        updatedQFunction(updatedQ)
        improvePolicy()
    }
}
