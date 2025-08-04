package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class BellmanQFunctionEstimator<State, Action>(
    private val gamma: Double,
    private val stateActionListProvider: StateActionListProvider<State, Action>
) : DPQFunctionEstimator<State, Action> {

    override fun estimate(
        q: QFunction<State, Action>,
        trajectory: ProbabilisticTrajectory<State, Action>
    ): QFunction<State, Action> {
        var updatedQ = q

        val grouped = trajectory.groupBy { it.state to it.action }

        for ((stateAction, transitions) in grouped) {
            val (s, a) = stateAction

            val expectedValue = transitions.sumOf { t ->
                val futureActions = stateActionListProvider(t.nextState)
                val maxQ = if (t.done || futureActions.isEmpty()) 0.0
                else futureActions.maxOf { aPrime -> q[t.nextState, aPrime] }

                t.probability * (t.reward + gamma * maxQ)
            }

            updatedQ = updatedQ.update(s, a, expectedValue)
        }

        return updatedQ
    }
}
