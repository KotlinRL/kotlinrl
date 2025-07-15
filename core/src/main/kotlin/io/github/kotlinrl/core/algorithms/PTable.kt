package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.MutablePolicy
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class PTable(
    vararg val shape: Int,
    private val defaultAction: Int = 0
): MutablePolicy<IntArray, Int> {
    private val policy: NDArray<Int, DN> = mk.dnarray<Int, DN>(shape) { defaultAction }.asDNArray()

    override operator fun get(state: IntArray): Int = policy[state]
    override operator fun set(state: IntArray, action: Int) {
        policy[state] = action
    }

    fun allStates(): List<IntArray> = cartesianProduct(*shape.map { 0 until it }.toTypedArray())

    override fun invoke(state: IntArray): Int = this[state]

    fun deepCopy(): PTable {
        val copy = PTable(*shape)
        for (s in allStates()) {
            copy[s] = this[s]
        }
        return copy
    }

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}