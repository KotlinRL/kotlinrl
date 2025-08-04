package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class QTableD4(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D3>, Int> {

    init {
        require(shape.size == 5) { "QTableD4 shape requires exactly 5 arguments" }
    }

    private val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    override fun get(state: NDArray<Int, D3>, action: Int): Double = base[state.asDNArray(), action]

    override fun update(
        state: NDArray<Int, D3>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, D3>, Int> {
        val updatedBase = base.update(state.asDNArray(), action, value) as QTableDN
        val new =
            QTableD4(shape = shape, deterministic = deterministic, tolerance = tolerance, defaultQValue = defaultQValue)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<NDArray<Int, D3>> =
        base.allStates().map { it.asD3Array() }

    override fun allActions(state: NDArray<Int, D3>): List<Int> =
        base.allActions(state.asDNArray())

    override fun maxValue(state: NDArray<Int, D3>): Double =
        base.maxValue(state.asDNArray())

    override fun bestAction(state: NDArray<Int, D3>): Int =
        base.bestAction(state.asDNArray())

    fun copy(): QTableD4 {
        val new = QTableD4(shape = shape, deterministic, tolerance, defaultQValue)
        base.copy().also { it.table.data.copyInto(new.base.table.data) }
        return new
    }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}
