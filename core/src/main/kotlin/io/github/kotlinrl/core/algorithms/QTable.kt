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
    vararg shape: Int,
): QFunction<IntArray, Int, NDArray<Double, D1>> {
    val dims: IntArray = shape
    private var table = mk.zeros<Double, DN>(dims, DoubleDataType).asDNArray()

    override operator fun get(state: IntArray, action: Int): Double = table[state + action]
    override operator fun set(state: IntArray, action: Int, value: Double) {
        table[state + action] = value
    }

    override fun qValues(index: IntArray): NDArray<Double, D1> {
        val axes = IntArray(index.size) { it }
        return table.view(index, axes).asDNArray().asD1Array()
    }

    override fun maxValue(state: IntArray): Double = qValues(state).max() ?: 0.0

    override fun bestAction(state: IntArray): Int = qValues(state).argMax()

    override fun save(path: String) {
        val d2 = table.reshape(dims.dropLast(1).reduce(Int::times), dims.last()).asD2Array()
        mk.write(File(path), d2)
    }

    fun copy(): QTable {
        val copy = QTable(*dims)
        table.data.copyInto(copy.table.data)
        return copy
    }

    override fun load(path: String) {
        val d2 = mk.read<Double, D2>(File(path))
        table =  when (dims.size) {
            2 -> d2.reshape(dims[0], dims[1])
            3 -> d2.reshape(dims[0], dims[1], dims[2])
            4 -> d2.reshape(dims[0], dims[1], dims[2], dims[3])
            else -> d2.reshape(dims[0], dims[1], dims[2], dims[3], *dims.copyOfRange(4, dims.size))
        }.asDNArray()
    }
}
