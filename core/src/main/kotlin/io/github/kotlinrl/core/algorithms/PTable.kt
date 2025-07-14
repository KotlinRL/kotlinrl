package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.MutablePolicy
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class PTable(
    private val gridSize: Int,
    private val defaultAction: Int = 0
): MutablePolicy<IntArray, Int> {
    private val policy: D2Array<Int> = mk.d2array(gridSize, gridSize) { defaultAction }

    override operator fun get(state: IntArray): Int = policy[state[0], state[1]]
    override operator fun set(state: IntArray, action: Int) {
        policy[state[0], state[1]] = action
    }

    fun allStates(): List<IntArray> = buildList {
        for (i in 0 until gridSize) {
            for (j in 0 until gridSize) {
                add(intArrayOf(i, j))
            }
        }
    }

    override fun invoke(state: IntArray): Int = this[state]
}