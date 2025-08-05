package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.base.TrajectoryQFunctionEstimator

class OffPolicyMonteCarloQFunctionEstimator<State, Action>(
    initTargetPolicy: QFunctionPolicy<State, Action>,
    private val behaviorPolicy: QFunctionPolicy<State, Action>,
    private val gamma: Double,
) : TrajectoryQFunctionEstimator<State, Action> {
    private val C: MutableMap<StateActionKey<*, *>, Double> = mutableMapOf()

    var targetPolicy: Policy<State, Action> = initTargetPolicy

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: Trajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        var G = 0.0
        var W = 1.0
        var currentQ = Q

        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G

            val key = stateActionKey(s, a)
            val oldC = C.getOrDefault(key, 0.0)
            val newC = oldC + W
            C[key] = newC

            val oldQ = currentQ[s, a]
            val updatedQ = oldQ + (W / newC) * (G - oldQ)
            currentQ = currentQ.update(s, a, updatedQ)

            if (a != targetPolicy(s)) break
            val prob = behaviorPolicy.probability(s, a)
            if (prob == 0.0) break
            W /= prob
        }

        return currentQ
    }
}
