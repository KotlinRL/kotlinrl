package io.github.kotlinrl.core.algorithms

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.io.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.*
import kotlin.collections.plus

class QTable(
    vararg val shape: Int,
): QFunction<IntArray, Int> {
    private var table = mk.zeros<Double, DN>(shape, DoubleDataType).asDNArray()

    override operator fun get(state: IntArray, action: Int): Double = table[state + action]
    override operator fun set(state: IntArray, action: Int, value: Double) {
        table[state + action] = value
    }

    private fun qValues(state: IntArray): NDArray<Double, D1> {
        val axes = IntArray(state.size) { it }
        return table.view(state, axes).asDNArray().asD1Array()
    }

    override fun maxValue(state: IntArray): Double = qValues(state).max() ?: 0.0

    override fun bestAction(state: IntArray): Int = qValues(state).argMax()

    override fun save(path: String) {
        if (shape.size == 1) {
            val d1 = table.reshape(shape[0]).asD1Array()
            mk.write(File(path), d1)
        } else {
            val d2 = table.reshape(shape.dropLast(1).reduce(Int::times), shape.last()).asD2Array()
            mk.write(File(path), d2)
        }
    }

    fun copy(): QTable {
        val copy = QTable(*shape)
        table.data.copyInto(copy.table.data)
        return copy
    }

    override fun load(path: String) {
        if (shape.size == 1) {
            val d1 = mk.read<Double, D1>(File(path))
            table = d1.reshape(shape[0]).asDNArray()
        } else {
            val d2 = mk.read<Double, D2>(File(path))
            table =  when (shape.size) {
                2 -> d2.reshape(shape[0], shape[1])
                3 -> d2.reshape(shape[0], shape[1], shape[2])
                4 -> d2.reshape(shape[0], shape[1], shape[2], shape[3])
                else -> d2.reshape(shape[0], shape[1], shape[2], shape[3], *shape.copyOfRange(4, shape.size))
            }.asDNArray()
        }
    }
}
