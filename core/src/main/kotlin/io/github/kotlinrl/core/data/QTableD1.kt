package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class QTableD1(
    vararg val shape: Int,
    private val deterministic: Boolean = true,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : EnumerableQFunction<Int, Int> {

    init {
        require(shape.size == 2) { "QTableD1 shape requires exactly 2 arguments" }
    }

    private val base = QTableDN(shape = shape, deterministic, tolerance, defaultQValue)

    override fun get(state: Int, action: Int): Double =
        base[mk.ndarray(intArrayOf(state)).asDNArray(), action]

    override fun update(
        state: Int,
        action: Int,
        value: Double
    ): EnumerableQFunction<Int, Int> =
        copy().also { it.base.table[intArrayOf(state) + action] = value }

    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }

    override fun maxValue(state: Int): Double =
        base.maxValue(mk.ndarray(intArrayOf(state)).asDNArray())

    override fun bestAction(state: Int): Int =
        base.bestAction(mk.ndarray(intArrayOf(state)).asDNArray())

    fun copy(): QTableD1 =
        QTableD1(
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

    fun asQTableD2(vararg shape: Int): QTableD2 =
        QTableD2(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun asQTableD3(vararg shape: Int): QTableD3 =
        QTableD3(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun asQTableD4(vararg shape: Int): QTableD4 =
        QTableD4(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun asQTableD5(vararg shape: Int): QTableD5 =
        QTableD5(
            shape = shape,
            deterministic = deterministic,
            tolerance = tolerance,
            defaultQValue = defaultQValue
        ).also {
            base.table.data.copyInto(it.base.table.data)
        }

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
