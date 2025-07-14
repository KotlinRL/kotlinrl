package io.github.kotlinrl.core.algorithms

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class PTable(
    private val gridSize: Int,
    private val defaultAction: Int = 0
) {
    private val policy: D2Array<Int> = mk.d2array(gridSize, gridSize) { defaultAction }

    operator fun get(state: IntArray): Int = policy[state[0], state[1]]
    operator fun set(state: IntArray, action: Int) {
        policy[state[0], state[1]] = action
    }

    fun allStates(): List<IntArray> = buildList {
        for (i in 0 until gridSize) {
            for (j in 0 until gridSize) {
                add(intArrayOf(i, j))
            }
        }
    }
}