package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.QFunction
import org.apache.commons.csv.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.*
import kotlin.math.*

class QTable(
    vararg val shape: Int,
    private val useArgMax: Boolean = false,
    private val tolerance: Double = 1e-6,
    private val defaultQValue: Double = 0.0
) : QFunction<IntArray, Int> {
    private var table: NDArray<Double, DN> = mk.dnarray<Double, DN>(shape) { defaultQValue }.asDNArray()

    override operator fun get(state: IntArray, action: Int): Double = table[state + action]
    override operator fun set(state: IntArray, action: Int, value: Double) {
        table[state + action] = value
    }

    private fun qValues(state: IntArray): NDArray<Double, D1> {
        val axes = IntArray(state.size) { it }
        return table.view(state, axes).asDNArray().asD1Array()
    }

    override fun maxValue(state: IntArray): Double = qValues(state).max() ?: 0.0

    override fun bestAction(state: IntArray): Int {
        val q = qValues(state)
        return if (useArgMax) {
            q.argMax()
        } else {
            val max = q.max() ?: 0.0
            val candidates = q.indices.filter { abs(q[it] - max) < tolerance }
            when {
                candidates.isNotEmpty() -> if (candidates.size > 1) candidates.random() else candidates.first()
                else -> q.indices.random()
            }
        }
    }

    fun copy(): QTable {
        val copy = QTable(*shape)
        table.data.copyInto(copy.table.data)
        return copy
    }

    override fun save(path: String) {
        mk.writeCsvSafely(path, table)
    }

    override fun load(path: String) {
        val dn = mk.readCsvSafely(path)
        val t = when (shape.size) {
            2 -> dn.reshape(shape[0], shape[1])
            3 -> dn.reshape(shape[0], shape[1], shape[2])
            4 -> dn.reshape(shape[0], shape[1], shape[2], shape[3])
            else -> dn.reshape(shape[0], shape[1], shape[2], shape[3], *shape.copyOfRange(4, shape.size))
        }.asDNArray()
        t.data.copyInto(table.data)
    }

    fun print() = println(table)
}

private fun mk.writeCsvSafely(path: String, ndarray: NDArray<Double, DN>): Unit =
    CSVFormat.DEFAULT.print(FileWriter(path)).use { printer ->
        val shape = ndarray.shape
        val dim = ndarray.dim
        when (dim) {
            D1 -> ndarray.forEach { printer.printRecord(it) }
            else -> {
                val numRows = shape.dropLast(1).reduce(Int::times)
                val rowLength = shape.last()

                val reshaped = ndarray.reshape(numRows, rowLength)
                for (i in 0 until numRows) {
                    val row = List(rowLength) { j -> reshaped[i, j] }
                    printer.printRecord(row)
                }
            }
        }
    }

private fun mk.readCsvSafely(path: String): NDArray<Double, DN> {
    val data = mutableListOf<DoubleArray>()

    CSVFormat.DEFAULT.parse(FileReader(path)).use { parser ->
        parser.records.forEach { record -> data.add(record.map { it.toDouble() }.toDoubleArray()) }
    }

    val colCount = data.firstOrNull()?.size ?: 0

    require(data.all { it.size == colCount }) {
        "Inconsistent number of columns in CSV"
    }

    return mk.ndarray(data.toTypedArray()).asDNArray()
}