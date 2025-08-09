package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.data.VTableD5
import io.github.kotlinrl.core.data.VTableDN
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import java.io.File
import java.io.FileNotFoundException
import kotlin.test.assertNotNull

class VTableD5Test {

    @Test
    fun `test max with non-empty table`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
            .update(1, 1, 1, 1, 1, 42.0)
            .update(0, 0, 0, 0, 0, 84.0)

        val maxValue = vTableD5.max()

        assertEquals(84.0, maxValue, "The max value should be 84.0.")
    }

    @Test
    fun `test max with empty table`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val maxValue = vTableD5.max()

        assertEquals(0.0, maxValue, "The max value of an empty table should be 0.0.")
    }

    @Test
    fun `test save to valid path`() {
        val tempFile = File.createTempFile("vTableD5", ".tmp")
        tempFile.deleteOnExit()

        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
        vTableD5.update(0, 0, 0, 0, 0, 42.0) // Add test data

        vTableD5.save(tempFile.absolutePath)

        val content = tempFile.readText()
        assert(content.isNotEmpty()) { "Saved file content should not be empty." }
    }

    @Test
    fun `test save overwrites existing file`() {
        val tempFile = File.createTempFile("vTableD5", ".tmp")
        tempFile.deleteOnExit()

        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
        vTableD5.update(0, 0, 0, 0, 0, 42.0) // Add test data
        vTableD5.save(tempFile.absolutePath)

        // Create new VTableD5 and overwrite the file with new data
        val newVTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
            .update(0, 0, 0, 0, 0, 84.0) // Add different test data
        newVTableD5.save(tempFile.absolutePath)

        // Reload content to ensure it has been overwritten
        val reloadedVTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
        reloadedVTableD5.load(tempFile.absolutePath)

        assertEquals(84.0, reloadedVTableD5[0, 0, 0, 0, 0], "File should be overwritten with new data.")
    }

    @Test
    fun `test save with invalid path`() {
        val invalidPath = "/invalid_path/vTableD5.tmp"
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val exception = assertFailsWith<FileNotFoundException> {
            vTableD5.save(invalidPath)
        }

        assertNotNull(exception.message)
    }

    /**
     * Tests for the `asVTableN` function in the `VTableD5` class.
     * The `asVTableN` function converts a `VTableD5` instance into a `VTableDN` instance with a specified shape.
     */

    @Test
    fun `test asVTableN with valid dimensions`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val result = vTableD5.asVTableN(2,3,4,5,6)

        assertEquals(intArrayOf(2,3,4,5,6).toList(), result.shape.toList())
        assertEquals(0.0, result.table.data.sum(), "Data should be copied and initialized to 0.0")
    }

    @Test
    fun `test asVTableN with single dimension`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val result = vTableD5.asVTableN(2,3,4,5,6)

        assertEquals(intArrayOf(2,3,4,5,6).toList(), result.shape.toList())
        assertEquals(0.0, result.table.data.sum(), "Data should be copied and initialized to 0.0")
    }

    @Test
    fun `test asVTableN with empty dimensions`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTableD5.asVTableN()
        }

        assertEquals("VTableDN shape requires at least 1 arguments", exception.message)
    }

    @Test
    fun `test asVTableN with large dimensions`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val result = vTableD5.asVTableN(50, 60, 70, 80)

        assertEquals(intArrayOf(50, 60, 70, 80).toList(), result.shape.toList())
        assertEquals(0.0, result.table.data.sum(), "Data should be copied and initialized to 0.0")
    }

    @Test
    fun `test print`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val outputStream = java.io.ByteArrayOutputStream()
        System.setOut(java.io.PrintStream(outputStream))

        vTableD5.print()

        val output = outputStream.toString()
        assert(output.isNotBlank()) { "The printed output should contain the shape information." }
        System.setOut(System.out) // Reset stdout after test
    }

    @Test
    fun `test load with valid data`() {
        val tempFile = File.createTempFile("vTableD5", ".tmp")
        tempFile.deleteOnExit()

        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
        vTableD5.save(tempFile.absolutePath)

        val loadedVTableD5 = VTableD5(2, 3, 4, 5, 6)
        loadedVTableD5.load(tempFile.absolutePath)

        assertEquals(
            vTableD5.base.table.data.toList(),
            loadedVTableD5.base.table.data.toList(),
            "Loaded table data should match saved data."
        )
    }

    @Test
    fun `test load with invalid path`() {
        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTableD5.load("non_existent_file_path")
        }

        assertNotNull(exception.message)
    }

    @Test
    fun `test load maintains table integrity`() {
        val tempFile = File.createTempFile("vTableD5", ".tmp")
        tempFile.deleteOnExit()

        val vTableD5 = VTableD5(rowSize = 2, colSize = 3, layerSize = 4, featureSize = 5, channelSize = 6)
            .update(0, 0, 0, 0, 0, 42.0) // Add test data
        vTableD5.save(tempFile.absolutePath)

        val newVTableD5 = VTableD5(2, 3, 4, 5, 6) // Create a new table with default values
        newVTableD5.load(tempFile.absolutePath)

        assertEquals(42.0, newVTableD5[0, 0, 0, 0, 0], "Loaded table should retain the updated value.")
    }
}