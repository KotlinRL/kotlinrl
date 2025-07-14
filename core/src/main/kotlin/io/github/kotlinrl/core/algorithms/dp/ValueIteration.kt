package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

class ValueIteration : Planner<IntArray, Int> {
    override fun plan(
        allStates: StateProvider<IntArray>,
        allActions: StateActionListProvider<IntArray, Int>,
        transition: TransitionFunction<IntArray, Int>,
        reward: RewardFunction<IntArray, Int>
    ): Policy<IntArray, Int> {
        val states = allStates()
        val vTable = VTable(states.size)
        val qTable = QTable(states.size, allActions(states[0]).size)
        TODO("Not yet implemented")
    }
}