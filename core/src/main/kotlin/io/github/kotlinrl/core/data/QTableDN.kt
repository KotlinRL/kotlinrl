package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.*

class QTableDN(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, DN>, Int> {

    init {
        require(shape.size >= 2) { "QTableDN shape requires at least 2 arguments" }
    }

    internal var table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { defaultQValue }.asDNArray()

    override operator fun get(state: NDArray<Int, DN>, action: Int): Double =
        table[state.toIntArray() + action]

    override fun update(
        state: NDArray<Int, DN>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, DN>, Int> =
        copy().also { it.table[state.toIntArray() + action] = value }

    override fun allStates(): List<NDArray<Int, DN>> {
        val stateShape = shape.dropLast(1) // all but action dimension
        val rawStates = cartesianProduct(*stateShape.map { 0 until it }.toTypedArray())
        return rawStates.map { mk.ndarray(it).asDNArray() }
    }

    private fun qValues(state: NDArray<Int, DN>): NDArray<Double, D1> {
        val axes = IntArray(state.shape[0]) { it }
        return table.view(state.toIntArray(), axes).asDNArray().asD1Array()
    }

    override fun maxValue(state: NDArray<Int, DN>): Double = qValues(state).max() ?: 0.0

    override fun bestAction(state: NDArray<Int, DN>): Int {
        val Q = qValues(state)
        return if (deterministic) {
            Q.argMax()
        } else {
            val max = Q.max() ?: 0.0
            val candidates = Q.indices.filter { abs(Q[it] - max) < tolerance }
            when {
                candidates.isNotEmpty() -> if (candidates.size > 1) candidates.random() else candidates.first()
                else -> Q.indices.random()
            }
        }
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

    fun copy(): QTableDN =
        QTableDN(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            table.data.copyInto(it.table.data)
        }

    private fun cartesianProduct(vararg ranges: Iterable<Int>): List<IntArray> {
        return ranges.fold(listOf(IntArray(0))) { acc, range ->
            acc.flatMap { prefix -> range.map { i -> prefix + i } }
        }
    }
}