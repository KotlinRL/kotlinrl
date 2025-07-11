package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.learn.QTable

class GreedyPolicy(
    private val qTable: QTable
) : Policy<IntArray, Int> {

    override operator fun invoke(state: IntArray): Int {
        return qTable.bestAction(state)
    }
}