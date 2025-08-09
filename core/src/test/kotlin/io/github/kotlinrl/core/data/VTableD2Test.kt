package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import kotlin.io.path.*
import kotlin.test.*

/**
 * Test class for the VTableD2 class.
 *
 * The VTableD2 class represents a two-dimensional value table that supports
 * enumerable operations. The `allStates` method returns a list of all possible
 * states represented as one-dimensional NDArray of integers.
 */
class VTableD2Test {

    @Test
    fun `allStates should return all states for a 2x2 table`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)
        val expectedStates = listOf(
            mk.ndarray(mk[0, 0]),
            mk.ndarray(mk[0, 1]),
            mk.ndarray(mk[1, 0]),
            mk.ndarray(mk[1, 1])
        )

        val resultStates = vTable.allStates()

        assertTrue(expectedStates.size == resultStates.size)
        expectedStates.forEachIndexed { index, expectedState ->
            assertTrue(expectedState.toComparable().compareTo(resultStates[index]) == 0)
        }
    }

    @Test
    fun `allStates should return no states when row or column size is zero`() {
        var exception = assertFailsWith<IllegalArgumentException> {
            VTableD2(rowSize = 0, colSize = 3)
        }
        assertEquals("Dimension 0 must be positive.", exception.message)

        exception = assertFailsWith<IllegalArgumentException> {
            VTableD2(rowSize = 3, colSize = 0)
        }
        assertEquals("Dimension 1 must be positive.", exception.message)
    }

    @Test
    fun `allStates should return correct states for larger tables`() {
        val vTable = VTableD2(rowSize = 3, colSize = 3)
        val expectedStates = listOf(
            mk.ndarray(mk[0, 0]),
            mk.ndarray(mk[0, 1]),
            mk.ndarray(mk[0, 2]),
            mk.ndarray(mk[1, 0]),
            mk.ndarray(mk[1, 1]),
            mk.ndarray(mk[1, 2]),
            mk.ndarray(mk[2, 0]),
            mk.ndarray(mk[2, 1]),
            mk.ndarray(mk[2, 2])
        )

        val resultStates = vTable.allStates()

        assertTrue(expectedStates.size == resultStates.size)
        expectedStates.forEachIndexed { index, expectedState ->
            assertTrue(expectedState.toComparable().compareTo(resultStates[index]) == 0)
        }
    }

    @Test
    fun `allStates should handle 1x1 table correctly`() {
        val vTable = VTableD2(rowSize = 1, colSize = 1)
        val expectedStates = listOf(
            mk.ndarray(mk[0, 0])
        )

        val resultStates = vTable.allStates()

        assertTrue(expectedStates.size == resultStates.size)
        expectedStates.forEachIndexed { index, expectedState ->
            assertTrue(expectedState.toComparable().compareTo(resultStates[index]) == 0)
        }
    }

    @Test
    fun `update should modify the value for specific row and column`() {
        val (row, col) = 0 to 1
        val value = 3.7
        val vTable = VTableD2(rowSize = 2, colSize = 2).update(row, col, value)

        assertEquals(value, vTable[row, col])
    }

    @Test
    fun `update should not affect other states`() {
        var vTable = VTableD2(rowSize = 2, colSize = 2)
        val value = 4.5

        vTable = vTable.update(0, 1, value)

        // Ensure other state remains default value
        assertEquals(0.0, vTable[1, 0])
    }

    @Test
    fun `update should handle multiple updates correctly`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        assertEquals(1.0, vTable[0, 0])
        assertEquals(2.0, vTable[0, 1])
        assertEquals(3.0, vTable[1, 0])
        assertEquals(4.0, vTable[1, 1])
    }

    @Test
    fun `max should return the highest value in a non-empty table`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 5.0)
            .update(1, 0, 3.0)
            .update(1, 1, 2.0)

        assertEquals(5.0, vTable.max())
    }

    @Test
    fun `max should return 0 for a table containing default values only`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)
        assertEquals(0.0, vTable.max())
    }

    @Test
    fun `max should correctly identify the maximum value after multiple updates`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 2.0)
            .update(0, 1, 4.0)
            .update(1, 0, 6.0)
            .update(1, 1, 8.0)

        assertEquals(8.0, vTable.max())
    }

    @Test
    fun `save should execute without errors`() {
        val mockPath = "test_vtable_save.txt"
        val vTable = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        assertDoesNotThrow { vTable.save(mockPath) }
        // Cleanup if necessary
        java.io.File(mockPath).delete()
    }

    @Test
    fun `save should persist correct data`() {
        val mockPath = "test_vtable_save_data.txt"
        val vTable = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 5.0)
            .update(0, 1, 6.0)
            .update(1, 0, 7.0)
            .update(1, 1, 8.0)

        vTable.save(mockPath)

        val savedContent = java.io.File(mockPath).readText()
        assertTrue(savedContent.contains("5.0"))
        assertTrue(savedContent.contains("6.0"))
        assertTrue(savedContent.contains("7.0"))
        assertTrue(savedContent.contains("8.0"))

        // Cleanup after test
        java.io.File(mockPath).delete()
    }

    @Test
    fun `load should successfully restore a previously saved table`() {
        val mockPath = "test_vtable_save_load.txt"
        val vTableSaved = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 9.0)
            .update(0, 1, 10.0)
            .update(1, 0, 11.0)
            .update(1, 1, 12.0)

        vTableSaved.save(mockPath)

        val vTableLoaded = VTableD2(rowSize = 2, colSize = 2)
        vTableLoaded.load(mockPath)

        assertEquals(9.0, vTableLoaded[0, 0])
        assertEquals(10.0, vTableLoaded[0, 1])
        assertEquals(11.0, vTableLoaded[1, 0])
        assertEquals(12.0, vTableLoaded[1, 1])

        // Cleanup after test
        java.io.File(mockPath).delete()
    }

    @Test
    fun `load should throw an exception for invalid file content`() {
        val mockPath = "test_vtable_invalid_file.txt"
        java.io.File(mockPath).writeText("invalid content")

        val vTable = VTableD2(rowSize = 2, colSize = 2)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTable.load(mockPath)
        }

        assertNotNull(exception.message)

        // Cleanup after test
        java.io.File(mockPath).delete()
    }

    @Test
    fun `load should handle an empty file gracefully`() {
        val mockPath = "test_vtable_empty_file.txt"
        val tempFile = createTempFile(mockPath)

        val vTable = VTableD2(rowSize = 2, colSize = 2)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTable.load(mockPath)
        }

        assertTrue(exception.message?.contains("File is empty") == true)
        tempFile.deleteIfExists()
    }

    @Test
    fun `print should display table correctly`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        val outputStream = java.io.ByteArrayOutputStream()
        System.setOut(java.io.PrintStream(outputStream))

        vTable.print()

        val expectedOutput = """
            [[1.0, 2.0],
            [3.0, 4.0]]
        """.trimIndent()

        assertEquals(expectedOutput, outputStream.toString().trim())
    }

    @Test
    fun `print should handle an empty table correctly`() {
        val vTable = VTableD2(rowSize = 2, colSize = 2)

        val outputStream = java.io.ByteArrayOutputStream()
        System.setOut(java.io.PrintStream(outputStream))

        vTable.print()

        val expectedOutput = """
            [[0.0, 0.0],
            [0.0, 0.0]]
        """.trimIndent()

        assertEquals(expectedOutput, outputStream.toString().trim())
    }

    @Test
    fun `asVTable3 should transform a 2D table into a 3D table and retain data`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        val vTableD3 = vTableD2.asVTable3(rowSize = 2, colSize = 2, layerSize = 1)

        assertEquals(1.0, vTableD3[0, 0, 0])
        assertEquals(2.0, vTableD3[0, 1, 0])
        assertEquals(3.0, vTableD3[1, 0, 0])
        assertEquals(4.0, vTableD3[1, 1, 0])
    }

    @Test
    fun `asVTable3 should throw an exception for invalid layer size`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTableD2.asVTable3(rowSize = 2, colSize = 2, layerSize = 0)
        }

        assertNotNull(exception.message)
    }

    @Test
    fun `asVTable4 should transform a 2D table into a 4D table and retain data`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        val vTableD4 = vTableD2.asVTable4(rowSize = 2, colSize = 2, layerSize = 1, featureSize = 1)

        assertEquals(1.0, vTableD4[0, 0, 0, 0])
        assertEquals(2.0, vTableD4[0, 1, 0, 0])
        assertEquals(3.0, vTableD4[1, 0, 0, 0])
        assertEquals(4.0, vTableD4[1, 1, 0, 0])
    }

    @Test
    fun `asVTable4 should throw an exception for invalid feature size`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTableD2.asVTable4(rowSize = 2, colSize = 2, layerSize = 1, featureSize = 0)
        }

        assertNotNull(exception.message)
    }

    @Test
    fun `asVTable5 should transform a 2D table into a 5D table and retain data`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        val vTableD5 = vTableD2.asVTable5(rowSize = 2, colSize = 2, layerSize = 1, featureSize = 1, channelSize = 1)

        assertEquals(1.0, vTableD5[0, 0, 0, 0, 0])
        assertEquals(2.0, vTableD5[0, 1, 0, 0, 0])
        assertEquals(3.0, vTableD5[1, 0, 0, 0, 0])
        assertEquals(4.0, vTableD5[1, 1, 0, 0, 0])
    }

    @Test
    fun `asVTable5 should throw an exception for invalid channel size`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)

        val exception = assertFailsWith<IllegalArgumentException> {
            vTableD2.asVTable5(rowSize = 2, colSize = 2, layerSize = 1, featureSize = 1, channelSize = 0)
        }

        assertNotNull(exception.message)
    }

    @Test
    fun `asVTableN should transform a 2D table into an N-dimensional table and retain data`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(1, 0, 3.0)
            .update(1, 1, 4.0)

        val shape = intArrayOf(2, 2, 1, 1, 1)
        val vTableDN = vTableD2.asVTableN(*shape)

        assertEquals(1.0, vTableDN[mk.ndarray(mk[0, 0, 0, 0, 0]).asDNArray()])
        assertEquals(2.0, vTableDN[mk.ndarray(mk[0, 1, 0, 0, 0]).asDNArray()])
        assertEquals(3.0, vTableDN[mk.ndarray(mk[1, 0, 0, 0, 0]).asDNArray()])
        assertEquals(4.0, vTableDN[mk.ndarray(mk[1, 1, 0, 0, 0]).asDNArray()])
    }

    @Test
    fun `asVTableN should throw an exception for invalid shape`() {
        val vTableD2 = VTableD2(rowSize = 2, colSize = 2)

        val invalidShape = intArrayOf(2, 2, 0) // Invalid due to zero dimension

        val exception = assertFailsWith<IllegalArgumentException> {
            vTableD2.asVTableN(*invalidShape)
        }

        assertNotNull(exception.message)
    }
}