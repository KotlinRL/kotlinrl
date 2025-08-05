package io.github.kotlinrl.core.data

import org.apache.commons.csv.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.*


/**
 * Writes the contents of the provided NDArray to a CSV file at the specified path.
 * Supports 1-dimensional and multi-dimensional arrays. For multi-dimensional arrays,
 * the array is reshaped appropriately into rows and written to the CSV file.
 *
 * @param path The file path where the CSV file will be written.
 * @param ndarray The NDArray containing the data to be written, with Double as the data type.
 * @return Unit, indicating that the method performs an operation without returning a value.
 */
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

/**
 * Reads a CSV file from the given file path and safely converts its data into a two-dimensional NDArray
 * of type Double. Ensures consistent column counts across all rows in the CSV.
 *
 * @param path The file path to the CSV file to be read.
 * @return A two-dimensional NDArray of type Double containing the data from the CSV file.
 * @throws IllegalArgumentException If the CSV contains inconsistent column counts across rows.
 * @throws IOException If an error occurs while reading the file.
 */
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
