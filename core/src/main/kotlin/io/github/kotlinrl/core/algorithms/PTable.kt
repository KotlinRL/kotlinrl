package io.github.kotlinrl.core.algorithms

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class PTable(
    private val size: Int,
    private val defaultAction: Int = 0
) {
    private val policy: D2Array<Int> = mk.d2array(size, size) { defaultAction }

    operator fun get(state: IntArray): Int = policy[state[0], state[1]]
    operator fun set(state: IntArray, action: Int) {
        policy[state[0], state[1]] = action
    }

    fun allStates(): Sequence<IntArray> = sequence {
        for (i in 0 until size) {
            for (j in 0 until size) {
                yield(intArrayOf(i, j))
            }
        }
    }
}