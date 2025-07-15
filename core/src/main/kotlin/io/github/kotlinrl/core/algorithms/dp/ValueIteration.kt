package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class ValueIteration(
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6
) : Planner<IntArray, Int> {
    override fun plan(
        vararg stateShape: Int,
        stateActionListProvider: StateActionListProvider<IntArray, Int>,
        transitionFunction: TransitionFunction<IntArray, Int>,
        rewardFunction: RewardFunction<IntArray, Int>
    ): Policy<IntArray, Int> {
        val vTable = VTable(*stateShape)
        val states = vTable.allStates()

        do {
            var delta = 0.0
            for (s in states) {
                val oldV = vTable[s]
                val bestValue = stateActionListProvider(s).maxOfOrNull { a ->
                    val next = transitionFunction(s, a)
                    val r = rewardFunction(s, a)
                    r + gamma * vTable[next]
                } ?: 0.0

                vTable[s] = bestValue
                delta = max(delta, abs(oldV - bestValue))
            }
        } while (delta > theta)

        val pi = PTable(*stateShape)
        for (s in states) {
            val bestAction = stateActionListProvider(s)
                .sorted()
                .maxByOrNull { a ->
                    val next = transitionFunction(s, a)
                    val r = rewardFunction(s, a)
                    r + gamma * vTable[next]
                } ?: error("No actions available for state ${s.contentToString()}")
            pi[s] = bestAction
        }
        return Policy { pi[it] }
    }
}