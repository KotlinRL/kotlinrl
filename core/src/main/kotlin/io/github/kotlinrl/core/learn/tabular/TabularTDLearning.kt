package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.QTable

abstract class TabularTDLearning(
    protected val qTable: QTable,
    protected val alpha: Double,
    protected val gamma: Double
) : TrajectoryObserver<IntArray, Int>, StateActionCallback<IntArray, Int> {
    protected var action: Int? = null

    override fun after(state: IntArray, action: Int) {
        this.action = action
    }
}
