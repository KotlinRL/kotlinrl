package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OffPolicyMonteCarloQFunctionEstimator<State, Action>(
    currentTargetPolicy: Policy<State, Action>,
    private val behaviorPolicy: StochasticPolicy<State, Action>,
    private val gamma: Double,
    private val stateActionKeyFunction: StateActionKeyFunction<State, Action>,
) : MonteCarloQFunctionEstimator<State, Action> {
    private val C: MutableMap<io.github.kotlinrl.core.algorithms.StateActionKey<*, *>, Double> = mutableMapOf()

    var targetPolicy: Policy<State, Action> = currentTargetPolicy

    override fun estimate(q: QFunction<State, Action>, trajectory: Trajectory<State, Action>, episode: Int): QFunction<State, Action> {
        var G = 0.0
        var W = 1.0
        var currentQ = q

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G

            val key = stateActionKeyFunction(s, a)
            val oldC = C.getOrDefault(key, 0.0)
            val newC = oldC + W
            C[key] = newC

            val oldQ = currentQ[s, a]
            val updatedQ = oldQ + (W / newC) * (G - oldQ)
            currentQ = currentQ.update(s, a, updatedQ)

            if (a != targetPolicy(s)) break
            val prob = behaviorPolicy(s, a)
            if (prob == 0.0) break
            W /= prob
        }

        return currentQ
    }
}
