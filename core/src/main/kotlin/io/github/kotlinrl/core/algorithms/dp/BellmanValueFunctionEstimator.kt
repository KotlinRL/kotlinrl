package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class BellmanValueFunctionEstimator<State, Action>(
    private val gamma: Double = 0.99
) : DPValueFunctionEstimator<State, Action> {

    override fun estimate(
        v: EnumerableValueFunction<State>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableValueFunction<State> {
        var newV = v

        val states = trajectory.map { it.state }.distinct()
        for (s in states) {
            val transitionsFromS = trajectory.filter { it.state == s }

            val newValue = transitionsFromS.sumOf { transition ->
                val r = transition.reward
                val sPrime = transition.nextState
                val p = transition.probability
                val done = transition.done

                val value = if (done) 0.0 else v[sPrime]
                p * (r + gamma * value)
            }

            newV = newV.update(s, newValue)
        }

        return newV
    }
}
