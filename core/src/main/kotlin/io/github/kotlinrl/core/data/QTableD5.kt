package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*

class QTableD5(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<NDArray<Int, D4>, Int> {

    init {
        require(shape.size == 6) { "QTableD5 shape requires exactly 6 arguments" }
    }

    internal val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    override fun get(state: NDArray<Int, D4>, action: Int): Double = base[state.asDNArray(), action]

    override fun update(
        state: NDArray<Int, D4>,
        action: Int,
        value: Double
    ): EnumerableQFunction<NDArray<Int, D4>, Int> =
        copy().also { it.base.table[state.toIntArray() + action] = value }


    override fun allStates(): List<NDArray<Int, D4>> =
        base.allStates().map { it.asD4Array() }

    override fun maxValue(state: NDArray<Int, D4>): Double =
        base.maxValue(state.asDNArray())

    override fun bestAction(state: NDArray<Int, D4>): Int =
        base.bestAction(state.asDNArray())

    fun copy(): QTableD5 =
        QTableD5(
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

    fun asQTableDN(vararg shape: Int): QTableDN =
        QTableDN(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.table.data)
        }
}
