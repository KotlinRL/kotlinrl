package io.github.kotlinrl.core.learn

import io.github.kotlinrl.core.policy.*

fun <State, Action> mutableQFunction(
    shape: IntArray,
    indexFor: (List<Any>, Int) -> IntArray,
    actionListProvider: StateActionListProvider<List<Any>, Int>
) {
    val qTable = QTable(shape, indexFor, actionListProvider)
    val state = listOf(20, 10, 1)
    val action = 0
    qTable[state, action] = 10.0
    val newValue = qTable[state, action]
    val max = qTable.maxValue(state)
    val best = qTable.bestAction(state)
}
