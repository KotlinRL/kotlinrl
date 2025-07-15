package io.github.kotlinrl.core.algorithms

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTable(
    vararg val shape: Int
) {
    private val table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { 0.0 }.asDNArray()

    fun max(): Double = table.data.max()

    operator fun get(state: IntArray): Double = table[state]

    operator fun set(state: IntArray, value: Double) {
        table[state] = value
    }

    fun allStates(): List<IntArray> = cartesianProduct(*shape.map { 0 until it }.toTypedArray())

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}
