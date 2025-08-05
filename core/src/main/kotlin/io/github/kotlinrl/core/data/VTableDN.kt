package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

class VTableDN(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, DN>> {

    init {
        require(shape.size >= 2) { "VTableDN shape requires at least 2 arguments" }
    }

    internal val table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { 0.0 }.asDNArray()

    override operator fun get(state: NDArray<Int, DN>): Double = table[state.toIntArray()]

    override fun update(
        state: NDArray<Int, DN>,
        value: Double
    ): EnumerableValueFunction<NDArray<Int, DN>> = copy().also { it.table[state.toIntArray()] = value }

    override fun max(): Double = table.data.max()

    override fun allStates(): List<NDArray<Int, DN>> {
        val rawStates = cartesianProduct(*shape.map { 0 until it }.toTypedArray())
        return rawStates.map { mk.ndarray(it).asDNArray() }
    }

    fun copy(): VTableDN {
        return VTableDN(*shape).also { table.data.copyInto(it.table.data) }
    }

    fun save(path: String) {
        mk.writeCsvSafely(path, table)
    }

    @Suppress("DuplicatedCode")
    fun load(path: String) {
        val dn = mk.readCsvSafely(path)
        val reshaped = when (shape.size) {
            2 -> dn.reshape(shape[0], shape[1])
            3 -> dn.reshape(shape[0], shape[1], shape[2])
            4 -> dn.reshape(shape[0], shape[1], shape[2], shape[3])
            else -> dn.reshape(shape[0], shape[1], shape[2], shape[3], *shape.copyOfRange(4, shape.size))
        }.asDNArray()
        reshaped.data.copyInto(table.data)
    }

    fun print() = println(table)

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}
