package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class ValueIteration : Planner<IntArray, Int> {
    override fun plan(
        size: Int,
        goal: IntArray,
        allActions: StateActionListProvider<IntArray, Int>,
        transition: TransitionFunction<IntArray, Int>,
        reward: RewardFunction<IntArray, Int>
    ): Policy<IntArray, Int> {
        val vTable = VTable(size)
        val states = vTable.allStates()

        val gamma = 0.99
        val theta = 1e-6

        do {
            var delta = 0.0
            for (s in states) {
                val oldV = vTable[s]
                val bestValue = allActions(s).maxOfOrNull { a ->
                    val next = transition(s, a)
                    val r = reward(s, a)
                    r + gamma * vTable[next]
                } ?: 0.0

                vTable[s] = bestValue
                delta = max(delta, abs(oldV - bestValue))
            }
        } while (delta > theta)

        val pi = PTable(size)
        for (s in states) {
            val bestAction = allActions(s)
                .sorted()
                .maxByOrNull { a ->
                    val next = transition(s, a)
                    val r = reward(s, a)
                    r + gamma * vTable[next]
                } ?: error("No actions available for state ${s.contentToString()}")
            pi[s] = bestAction
        }
        return Policy { pi[it] }
    }
}