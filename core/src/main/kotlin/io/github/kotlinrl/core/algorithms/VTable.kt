package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.ValueFunction
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTable(
    vararg val shape: Int
): ValueFunction<IntArray> {
    private val table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { 0.0 }.asDNArray()

    override operator fun get(state: IntArray): Double = table[state]
    override operator fun set(state: IntArray, value: Double) {
        table[state] = value
    }

    override fun max(): Double = table.data.max()

    override fun allStates(): List<IntArray> = cartesianProduct(*shape.map { 0 until it }.toTypedArray())

    fun copy(): VTable {
        val copy = VTable(*shape)
        table.data.copyInto(copy.table.data)
        return copy
    }

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}
