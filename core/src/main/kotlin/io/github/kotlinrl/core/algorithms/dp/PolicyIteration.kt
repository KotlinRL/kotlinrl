package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class PolicyIteration(
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    val vTable: VTable,
    val pTable: PTable
) : Planner<IntArray, Int> {

    override fun plan(
        stateActionListProvider: StateActionListProvider<IntArray, Int>,
        transitionFunction: TransitionFunction<IntArray, Int>
    ): Policy<IntArray, Int> {
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
                    val (next, r) = transitionFunction(s, a)
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
                    val (next, r) = transitionFunction(s, a)
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
