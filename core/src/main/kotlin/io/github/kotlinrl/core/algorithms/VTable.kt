package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTable(
    vararg val shape: Int
) : EnumerableValueFunction<IntArray> {
    private val table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { 0.0 }.asDNArray()

    override operator fun get(state: IntArray): Double = table[state]

    override fun update(
        state: IntArray,
        value: Double
    ): EnumerableValueFunction<IntArray> = copy().also { it.table[state] = value }

    override fun max(): Double = table.data.max()

    override fun allStates(): List<IntArray> = cartesianProduct(*shape.map { 0 until it }.toTypedArray())

    private fun copy(): VTable {
        return VTable(*shape).also { table.data.copyInto(it.table.data) }
    }

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}
