package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import io.github.kotlinrl.core.algorithms.PTable
import kotlin.math.*

class ValueIteration<State, Action>(
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6,
    val vTable: ValueFunction<State>,
    val pTable: MutablePolicy<State, Action>,
    private val actionComparator: Comparator<Action>
) : Planner<State, Action> {
    override fun plan(
        stateActionListProvider: StateActionListProvider<State, Action>,
        transitionFunction: TransitionFunction<State, Action>
    ): Policy<State, Action> {

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
                .sortedWith(actionComparator)
                .maxByOrNull { a ->
                    val (next, r) = transitionFunction(s, a)
                    r + gamma * vTable[next]
                } ?: error("No actions available for state $s")
            pTable[s] = bestAction
        }

        val pi = pTable.copy()
        return Policy { pi[it] }
    }
}
private fun Any?.format(): String = when (this) {
    is IntArray -> this.contentToString()
    is Array<*> -> this.contentToString()
    else -> this.toString()
}