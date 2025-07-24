package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

class QLearning<State, Action>(
    qTable: QFunction<State, Action>,
    alpha: ParameterSchedule,
    gamma: Double
) : TabularTDLearning<State, Action>(qTable, alpha, gamma) {

    override fun invoke(transition: Transition<State, Action>) {
        val (s, a, r, sPrime) = transition
        val done = transition.done

        val currentValue = qTable[s, a]
        val nextValue = if (done) 0.0 else qTable.maxValue(sPrime)
        val target = r + gamma * nextValue
        val updated = currentValue + alpha() * (target - currentValue)

        qTable[s, a] = updated
    }
}
