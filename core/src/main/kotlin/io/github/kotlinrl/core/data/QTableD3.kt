package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class QTableD3(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D2>, Int> {

    init {
        require(shape.size == 4) { "QTableD3 shape requires exactly 4 arguments" }
    }

    private val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    override fun get(state: NDArray<Int, D2>, action: Int): Double = base[state.asDNArray(), action]

    override fun update(
        state: NDArray<Int, D2>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, D2>, Int> {
        val updatedBase = base.update(state.asDNArray(), action, value) as QTableDN
        val new =
            QTableD3(shape = shape, deterministic = deterministic, tolerance = tolerance, defaultQValue = defaultQValue)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<NDArray<Int, D2>> =
        base.allStates().map { it.asD2Array() }

    override fun allActions(state: NDArray<Int, D2>): List<Int> =
        base.allActions(state.asDNArray())

    override fun maxValue(state: NDArray<Int, D2>): Double =
        base.maxValue(state.asDNArray())

    override fun bestAction(state: NDArray<Int, D2>): Int =
        base.bestAction(state.asDNArray())

    fun copy(): QTableD3 {
        val new = QTableD3(shape = shape, deterministic, tolerance, defaultQValue)
        base.copy().also { it.table.data.copyInto(new.base.table.data) }
        return new
    }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}
