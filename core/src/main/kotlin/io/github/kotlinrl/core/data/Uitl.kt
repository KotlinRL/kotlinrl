package io.github.kotlinrl.core.data

import org.apache.commons.csv.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.*


internal fun mk.writeCsvSafely(path: String, ndarray: NDArray<Double, DN>): Unit =
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

internal fun mk.readCsvSafely(path: String): NDArray<Double, DN> {
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
