package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

abstract class NStepTD<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val n: Int,
    private val estimator: TrajectoryQFunctionEstimator<State, Action>,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
) : TrajectoryQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate) {

    private val window = ArrayDeque<Transition<State, Action>>()
    private var episode = 0
    private var tailAction: Action? = null
        set(value) {
            field = value
            when(estimator) {
                is NStepTDQFunctionEstimator -> estimator.tailAction = value
            }
        }

    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        this.episode = episode
    }

    override fun observe(transition: Transition<State, Action>) {
        window.addLast(transition)

        if (!transition.done && window.size >= n + 1) {
            step(n)
        } else if(transition.done) {
            tailAction = null
            while(window.isNotEmpty()) {
                step(n)
            }
        } else {
            while (window.size > n) window.removeFirst()
        }
    }

    protected fun step(n: Int) {
        tailAction = window.elementAtOrNull(n)?.action
        window.removeFirst()
        super.observe(window.take(n), episode)
    }
}