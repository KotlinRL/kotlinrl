package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class BellmanQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val stateActions: StateActions<State, Action>
) : DPQFunctionEstimator<State, Action> {

    override fun estimate(
        Q: EnumerableQFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): EnumerableQFunction<State, Action> {
        var updatedQ = Q

        val grouped = trajectory.groupBy { it.state to it.action }

        for ((stateAction, transitions) in grouped) {
            val (s, a) = stateAction

            val expectedValue = transitions.sumOf { t ->
                val futureActions = stateActions(t.nextState)
                val maxQ = if (t.done || futureActions.isEmpty()) 0.0
                else futureActions.maxOf { aPrime -> Q[t.nextState, aPrime] }

                t.probability * (t.reward + gamma * maxQ)
            }

            updatedQ = updatedQ.update(s, a, expectedValue)
        }

        return updatedQ
    }
}
