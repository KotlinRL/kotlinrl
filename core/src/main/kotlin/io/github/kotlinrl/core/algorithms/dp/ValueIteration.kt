package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class ValueIteration<State, Action>(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    private val actionComparator: Comparator<Action>
) : DPIteration<State, Action>(gamma, theta, vTable, pTable) {
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
                    val (_, _, r, next) = transitionFunction(s, a)
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
                    val (_, _, r, next) = transitionFunction(s, a)
                    r + gamma * vTable[next]
                } ?: error("No actions available for state ${s.format()}")
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