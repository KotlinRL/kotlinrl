package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class SARSA<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double
) : TabularTDLearning<State, Action>(qTable, alpha, gamma), TrajectoryLearner<State, Action> {

    private var lastTransition: Transition<State, Action>? = null

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        lastTransition = null
    }

    override fun invoke(transition: Transition<State, Action>) {
        // If no previous transition, store and wait for next
        val prev = lastTransition
        lastTransition = transition

        if (prev == null) return

        val (s, a) = prev
        val (sPrime, aPrime, r) = transition

        val currentValue = qTable[s, a]
        val nextValue = if (transition.done) 0.0 else qTable[sPrime, aPrime]

        val target = r + gamma * nextValue
        val updated = currentValue + alpha() * (target - currentValue)

        qTable[s, a] = updated
    }
}



