package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.PTable
import kotlin.math.*

class ValueIteration(
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

        do {
            var delta = 0.0
            for (s in states) {
                val oldV = vTable[s]
                val bestValue = stateActionListProvider(s).maxOfOrNull { a ->
                    val (next, r) = transitionFunction(s, a)
                    r + gamma * vTable[next]
                } ?: 0.0

                vTable[s] = bestValue
                delta = max(delta, abs(oldV - bestValue))
            }
        } while (delta > theta)

        for (s in states) {
            val bestAction = stateActionListProvider(s)
                .sorted()
                .maxByOrNull { a ->
                    val (next, r) = transitionFunction(s, a)
                    r + gamma * vTable[next]
                } ?: error("No actions available for state ${s.contentToString()}")
            pTable[s] = bestAction
        }

        val pi = pTable.deepCopy()
        return Policy { pi[it] }
    }
}