package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*
import kotlin.math.*

class PolicyIteration(
    private val gamma: Double = 0.99,
    private val theta: Double = 1e-6
) : Planner<IntArray, Int> {

    override fun plan(
        stateShape: IntArray,
        allActions: StateActionListProvider<IntArray, Int>,
        transition: TransitionFunction<IntArray, Int>,
        reward: RewardFunction<IntArray, Int>
    ): Policy<IntArray, Int> {
        val vTable = VTable(*stateShape)
        val pi = PTable(*stateShape)
        val states = vTable.allStates()

        // Initial policy: arbitrary (e.g., all zeros)
        for (s in states) {
            val actions = allActions(s)
            pi[s] = actions.firstOrNull() ?: continue
        }

        var policyStable: Boolean
        do {
            // Policy Evaluation
            do {
                var delta = 0.0
                for (s in states) {
                    val oldV = vTable[s]
                    val a = pi[s]
                    val next = transition(s, a)
                    val r = reward(s, a)
                    val newV = r + gamma * vTable[next]
                    vTable[s] = newV
                    delta = max(delta, abs(oldV - newV))
                }
            } while (delta > theta)

            // Policy Improvement
            policyStable = true
            for (s in states) {
                val oldAction = pi[s]
                val bestAction = allActions(s).maxByOrNull { a ->
                    val next = transition(s, a)
                    val r = reward(s, a)
                    r + gamma * vTable[next]
                } ?: oldAction

                if (oldAction != bestAction) {
                    pi[s] = bestAction
                    policyStable = false
                }
            }

        } while (!policyStable)

        return Policy { pi[it] }
    }
}
