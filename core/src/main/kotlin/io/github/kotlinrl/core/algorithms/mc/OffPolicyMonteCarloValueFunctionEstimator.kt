package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.defaultStateKeyFunction

class OffPolicyMonteCarloValueFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val behaviorPolicy: StochasticPolicy<State, Action>,
    private var targetPolicy: Policy<State, Action>,
    private val stateKeyFunction: StateKeyFunction<State> = ::defaultStateKeyFunction
) : MonteCarloValueFunctionEstimator<State, Action> {

    private val C: MutableMap<Comparable<*>, Double> = mutableMapOf()

    override fun estimate(v: ValueFunction<State>, trajectory: Trajectory<State, Action>): ValueFunction<State> {
        var G = 0.0
        var W = 1.0
        var newV = v

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G
            val key = stateKeyFunction(s)

            val c = C.getOrDefault(key, 0.0) + W
            C[key] = c

            val oldV = newV[s]
            val newValue = oldV + (W / c) * (G - oldV)
            newV = newV.update(s, newValue)

            if (a != targetPolicy(s)) break

            val prob = behaviorPolicy(s, a)
            if (prob == 0.0) break
            W /= prob
        }

        return newV
    }
}
