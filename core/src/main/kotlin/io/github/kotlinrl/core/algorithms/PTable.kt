package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.MutablePolicy
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class PTable(
    vararg val stateDims: Int,
    private val defaultAction: Int = 0
): MutablePolicy<IntArray, Int> {
    private val policy: NDArray<Int, DN> = mk.dnarray(stateDims) { defaultAction }

    override operator fun get(state: IntArray): Int = policy[state[0], state[1]]
    override operator fun set(state: IntArray, action: Int) {
        policy[state[0], state[1]] = action
    }

    fun allStates(): List<IntArray> = cartesianProduct(*stateDims.map { 0 until it }.toTypedArray())

    override fun invoke(state: IntArray): Int = this[state]

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}