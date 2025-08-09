package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.*
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.test.Test
import kotlin.test.assertEquals

class QTableD5Test {

    @Test
    fun `asQTableDN produces QTableDN with correct shape and default values`() {
        val qTableD5 = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
        val qTableDN = qTableD5.asQTableDN(2, 2, 2, 2, 2, 3)

        assertEquals(intArrayOf(2, 2, 2, 2, 2, 3).toList(), qTableDN.shape.toList())
        qTableDN.allStates().forEach { state ->
            assertEquals(0.0, qTableDN[state, 0])
        }
    }

    @Test
    fun `asQTableDN copies action-values correctly`() {
        val qTableD5 = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 10.0)
            .update(1, 1, 1, 1, 1, 2, 5.0)

        val qTableDN = qTableD5.asQTableDN(2, 2, 2, 2, 2, 3)
        assertEquals(10.0, qTableDN[1, 1, 1, 1, 1, 0])
        assertEquals(5.0, qTableDN[1, 1, 1, 1, 1, 2])
    }

    @Test
    fun `asQTableDN produces an independent QTableDN object`() {
        var qTableD5 = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 10.0)

        val qTableDN = qTableD5.asQTableDN(2, 2, 2, 2, 2, 3)
        qTableD5 = qTableD5.update(1, 1, 1, 1, 1, 0, 20.0)

        assertEquals(10.0, qTableDN[1, 1, 1, 1, 1, 0])
        assertEquals(20.0, qTableD5[1, 1, 1, 1, 1, 0])
    }

    @Test
    fun `load restores QTableD5 correctly`() {
        val originalQTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 10.0)

        val tempFile = kotlin.io.path.createTempFile("qtable_test", ".tmp").toFile()
        tempFile.deleteOnExit()
        originalQTable.save(tempFile.absolutePath)

        val loadedQTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
        loadedQTable.load(tempFile.absolutePath)

        val restoredValue = loadedQTable[1, 1, 1, 1, 1, 0]
        assertEquals(10.0, restoredValue)
    }

    @Test
    fun `load handles invalid file gracefully`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)

        val exception = kotlin.test.assertFailsWith<Exception> {
            qTable.load("nonexistent_file.tmp")
        }

        assertTrue(exception.message?.contains("File does not exist") == true)
    }

    @Test
    fun `bestAction returns the correct action for a state with predefined Q-values`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 5.0)
            .update(1, 1, 1, 1, 1, 1, 15.0)
            .update(1, 1, 1, 1, 1, 2, 10.0)

        val bestAction = qTable.bestAction(1, 1, 1, 1, 1)

        assertEquals(1, bestAction)
    }

    @Test
    fun `bestAction handles multiple actions with the same maximum value`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 0, 0, 10.0)
            .update(0, 0, 0, 0, 0, 1, 11.0)

        val bestAction = qTable.bestAction(0, 0, 0, 0, 0)

        assertEquals(1, bestAction)
    }

    @Test
    fun `bestAction respects default Q value for uninitialized states`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = -1.0)

        val bestAction = qTable.bestAction(0, 0, 0, 0, 0)

        assertEquals(0, bestAction) // Default to the first action since all have the same value.
    }

    @Test
    fun `bestAction respects the deterministic flag`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, deterministic = false, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 10.0)
            .update(1, 1, 1, 1, 1, 1, 20.0)
            .update(1, 1, 1, 1, 1, 2, 15.0)

        val bestAction = qTable.bestAction(1, 1, 1, 1, 1)

        assertEquals(1, bestAction) // Even with non-deterministic mode, best action should match maximum value.
    }

    @Test
    fun `toV returns a VTableD5 with correct shape`() {
        val qTable = QTableD5(3, 3, 3, 3, 3, 4, defaultQValue = 1.0)
        val vTable = qTable.toV()

        assertEquals(5, vTable.shape.size)
        assertEquals(3, vTable.shape[0])
        assertEquals(3, vTable.shape[1])
        assertEquals(3, vTable.shape[2])
        assertEquals(3, vTable.shape[3])
    }

    @Test
    fun `toV converts Q-values to V-values correctly`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 0, 0, 10.0)
            .update(0, 0, 0, 0, 0, 1, 5.0)
            .update(0, 0, 0, 0, 0, 2, 15.0)

        val vTable = qTable.toV()

        val expectedValue = 15.0

        assertEquals(expectedValue, vTable[mk.ndarray(mk[mk[mk[mk[0, 0, 0, 0, 0]]]])])
    }

    @Test
    fun `toV handles default Q values correctly`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = -1.0)
        val vTable = qTable.toV()

        val expectedValue = -1.0

        assertEquals(expectedValue, vTable[mk.ndarray(mk[mk[mk[mk[0, 0, 0, 0, 0]]]])])
    }

    @Test
    fun `toV handles deterministic flag correctly`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, deterministic = false, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 1, 20.0)

        val vTable = qTable.toV()

        val expectedValue = 20.0

        assertEquals(expectedValue, vTable[1, 1, 1, 1, 1])
    }

    @Test
    fun `toV operates on empty QTableD5 correctly`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
        val vTable = qTable.toV()

        val expectedValue = 0.0

        vTable.allStates().forEach { state ->
            assertEquals(expectedValue, vTable[state])
        }
    }

    @Test
    fun `maxValue returns the highest action-value for a state`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 5.0)
            .update(1, 1, 1, 1, 1, 1, 15.0)
            .update(1, 1, 1, 1, 1, 2, 10.0)

        val maxValue = qTable.maxValue(1, 1, 1, 1, 1)

        assertEquals(15.0, maxValue)
    }

    @Test
    fun `maxValue returns default value for uninitialized states`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = -2.0)

        val maxValue = qTable.maxValue(0, 0, 0, 0, 0)

        assertEquals(-2.0, maxValue)
    }

    @Test
    fun `maxValue works correctly for the deterministic flag`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, deterministic = false, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 0, 0, 12.0)
            .update(0, 0, 0, 0, 0, 1, 7.0)
            .update(0, 0, 0, 0, 0, 2, 25.0)

        val maxValue = qTable.maxValue(0, 0, 0, 0, 0)

        assertEquals(25.0, maxValue)
    }

    @Test
    fun `save saves to file correctly`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = -1.0)
            .update(0, 0, 0, 0, 0, 0, 5.0)

        val tempFile = kotlin.io.path.createTempFile("qtable_test", ".tmp").toFile()
        tempFile.deleteOnExit()

        qTable.save(tempFile.absolutePath)

        assertTrue(tempFile.exists(), "The file should have been created.")
        assertTrue(tempFile.length() > 0, "The file should not be empty.")
    }

    @Test
    fun `print outputs QTableD5 correctly`() {
        val qTable = QTableD5(2, 2, 2, 2, 2, 3, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 1, 0, 5.0)
            .update(1, 1, 1, 1, 1, 1, 15.0)
            .update(1, 1, 1, 1, 1, 2, 10.0)

        val outputStream = java.io.ByteArrayOutputStream()
        val printStream = java.io.PrintStream(outputStream)
        System.setOut(printStream)

        qTable.print()
        System.setOut(System.out)

        val output = outputStream.toString()
        assertTrue(output.contains("5.0"))
        assertTrue(output.contains("15.0"))
        assertTrue(output.contains("10.0"))
    }
}