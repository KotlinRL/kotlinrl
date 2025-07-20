package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class PolicyIteration<State, Action>(
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    val vTable: ValueFunction<State>,
    val pTable: MutablePolicy<State, Action>
) : Planner<State, Action> {

    override fun plan(
        stateActionListProvider: StateActionListProvider<State, Action>,
        transitionFunction: TransitionFunction<State, Action>
    ): Policy<State, Action> {
        val states = vTable.allStates()

        // Initial policy: arbitrary (e.g., all zeros)
        for (s in states) {
            val actions = stateActionListProvider(s)
            pTable[s] = actions.firstOrNull() ?: continue
        }

        var policyStable: Boolean
        do {
            // Policy Evaluation
            do {
                var delta = 0.0
                for (s in states) {
                    val oldV = vTable[s]
                    val a = pTable[s]
                    val (_, _, r, next) = transitionFunction(s, a)
                    val newV = r + gamma * vTable[next]
                    vTable[s] = newV
                    delta = max(delta, abs(oldV - newV))
                }
            } while (delta > theta)

            // Policy Improvement
            policyStable = true
            for (s in states) {
                val oldAction = pTable[s]
                val bestAction = stateActionListProvider(s).maxByOrNull { a ->
                    val (_, _, r, next) = transitionFunction(s, a)
                    r + gamma * vTable[next]
                } ?: oldAction

                if (oldAction != bestAction) {
                    pTable[s] = bestAction
                    policyStable = false
                }
            }

        } while (!policyStable)

        val pi = pTable.copy()
        return Policy { pi[it] }
    }
}
