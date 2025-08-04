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

    override fun update(state: Int, action: Int, value: Double): EnumerableQFunction<Int, Int> {
        val updatedBase = base.update(
            state = mk.ndarray(intArrayOf(state)).asDNArray(),
            action,
            value
        ) as QTableDN

        val new =
            QTableD1(shape = shape, deterministic = deterministic, tolerance = tolerance, defaultQValue = defaultQValue)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }


    override fun allActions(state: Int): List<Int> =
        base.allActions(mk.ndarray(intArrayOf(state)).asDNArray())


    override fun maxValue(state: Int): Double =
        base.maxValue(mk.ndarray(intArrayOf(state)).asDNArray())

    override fun bestAction(state: Int): Int =
        base.bestAction(mk.ndarray(intArrayOf(state)).asDNArray())

    fun copy(): QTableD1 {
        val new = QTableD1(shape = shape, deterministic, tolerance, defaultQValue)
        base.copy().also { it.table.data.copyInto(new.base.table.data) }
        return new
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
}
