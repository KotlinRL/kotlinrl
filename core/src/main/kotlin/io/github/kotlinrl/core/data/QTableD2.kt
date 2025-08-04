package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

class QTableD2(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D1>, Int> {

    init {
        require(shape.size == 3) { "QTableD2 shape requires exactly 3 arguments" }
    }

    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    override fun get(state: NDArray<Int, D1>, action: Int): Double = base[state.asDNArray(), action]

    override fun update(
        state: NDArray<Int, D1>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, D1>, Int> =
        copy().also { it.base.table[state.toIntArray() + action] = value }

    override fun allStates(): List<NDArray<Int, D1>> =
        base.allStates().map { it.asD1Array() }

    override fun allActions(state: NDArray<Int, D1>): List<Int> =
        base.allActions(state.asDNArray())

    override fun maxValue(state: NDArray<Int, D1>): Double =
        base.maxValue(state.asDNArray())

    override fun bestAction(state: NDArray<Int, D1>): Int =
        base.bestAction(state.asDNArray())

    fun copy(): QTableD2 =
        QTableD2(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}
