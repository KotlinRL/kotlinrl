package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

class OffPolicyMonteCarloControl<State, Action>(
    private val qTable: QFunction<State, Action>,
    private val gamma: Double,
    private val targetPolicy: Policy<State, Action>,
    private val probability: ProbabilityFunction<State, Action>,
    private val stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultKeyFunction
) : TrajectoryLearner<State, Action> {

    private val C: MutableMap<StateActionKey<*, *>, Double> = mutableMapOf()

    override fun invoke(trajectory: Trajectory<State, Action>, episode: Int) {
        var G = 0.0
        var W = 1.0
        for ((s, a, r) in trajectory.asReversed()) {
            G = r + gamma * G

            val key = stateActionKeyFunction(s, a)
            val oldC = C.getOrDefault(key, 0.0)
            val newC = oldC + W
            C[key] = newC

            val q = qTable[s, a]
            qTable[s, a] = q + (W / newC) * (G - q)

            if (a != targetPolicy(s)) break
            val prob = probability(s, a)
            if (prob == 0.0) break
            W /= prob
        }
    }
}