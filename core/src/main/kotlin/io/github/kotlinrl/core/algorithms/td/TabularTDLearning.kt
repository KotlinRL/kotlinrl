package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.algorithms.QTable

abstract class TabularTDLearning(
    protected val qTable: QTable,
    protected val alpha: Double,
    protected val gamma: Double
) : TrajectoryObserver<IntArray, Int> {

}
