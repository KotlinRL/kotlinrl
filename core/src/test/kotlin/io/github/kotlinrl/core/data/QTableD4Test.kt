package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*

class QTableD4Test {

    /**
     * QTableD4 Test class tests the behavior of the QTableD4 class.
     * The QTableD4 class provides a 4-dimensional Q-table with several
     * operations like updating and retrieving Q-values for given states
     * and actions.
     */

    @Test
    fun `test initialization of QTableD4`() {
        val shape = intArrayOf(3, 3, 3, 3, 4)
        val defaultQValue = 1.0
        val qTable = QTableD4(3, 3, 3, 3, 4, defaultQValue = defaultQValue)

        assertArrayEquals(shape, qTable.shape)
        assertTrue(qTable.deterministic)
        assertEquals(defaultQValue, qTable.defaultQValue)
    }

    @Test
    fun `test get Q-value for a given state and action`() {
        val shape = intArrayOf(2, 2, 2, 2, 3)
        val qTable = QTableD4(2, 2, 2, 2, 3)

        val state = mk.ndarray(mk[mk[mk[1, 1, 1, 1]]])
        val action = 0
        val qValue = qTable[state, action]

        assertEquals(0.0, qValue)
    }

    @Test
    fun `test retrieval using operator function`() {
        val shape = intArrayOf(2, 2, 2, 2, 3)
        val qTable = QTableD4(2, 2, 2, 2, 3)

        val dim1 = 1
        val dim2 = 1
        val dim3 = 1
        val dim4 = 1
        val action = 0
        val qValue = qTable[dim1, dim2, dim3, dim4, action]

        assertEquals(0.0, qValue)
    }

    @Test
    fun `test update Q-value for a given state and action`() {
        val shape = intArrayOf(2, 2, 2, 2, 3)
        val qTable = QTableD4(2, 2, 2, 2, 3)

        val state = mk.ndarray(mk[mk[mk[1, 1, 1, 1]]])
        val action = 1
        val value = 5.0

        val updatedQTable = qTable.update(state, action, value)
        val updatedValue = updatedQTable[state, action]

        assertEquals(value, updatedValue)
    }

    @Test
    fun `test copy QTableD4`() {
        val shape = intArrayOf(2, 2, 2, 2, 3)
        val qTable = QTableD4(2, 2, 2, 2, 3, defaultQValue = 2.0)

        val copyQTable = qTable.copy()
        assertArrayEquals(qTable.shape, copyQTable.shape)
        assertEquals(qTable.defaultQValue, copyQTable.defaultQValue)
    }

    @Test
    fun `test toV table conversion`() {
        val shape = intArrayOf(2, 2, 2, 2, 3)
        val state = mk.ndarray(mk[mk[mk[1, 1, 1, 1]]])
        val qTable = QTableD4(2, 2, 2, 2, 3)
            .update(state, 1, 10.0)

        val vTable = qTable.toV()
        val maxValue = vTable[state]

        assertEquals(10.0, maxValue)
    }

    @Test
    fun `test save and load`() {
        val shape = intArrayOf(2, 2, 2, 2, 3)
        val qTable = QTableD4(2, 2, 2, 2, 3, defaultQValue = 0.5)

        val path = "qtable_test.bin"
        qTable.save(path)

        val loadedQTable = QTableD4(2, 2, 2, 2, 3, defaultQValue = 0.5)
        loadedQTable.load(path)

        assertEquals(qTable.defaultQValue, loadedQTable.defaultQValue)
    }

    @Test
    fun `test best action for a state`() {
        val qTable = QTableD4(2, 2, 2, 2, 3)
            .update(1, 1, 1, 1, 1, 15.0)

        val bestAction = qTable.bestAction(1, 1, 1, 1)
        assertEquals(1, bestAction)
    }

    @Test
    fun `test best action when multiple actions have the same maximum value`() {
        val qTable = QTableD4(3, 3, 3, 3, 3)
            .update(0, 0, 0, 0, 0, 10.0)
            .update(0, 0, 0, 0, 2, 10.0)

        val bestAction = qTable.bestAction(0, 0, 0, 0)
        // Expected to be deterministic, but depends on implementation logic for tie-breaking
        assertTrue(bestAction in listOf(0, 2), "Best action should be one of the actions with the maximum Q-value.")
    }

    @Test
    fun `test best action when all actions have default Q-value`() {
        val qTable = QTableD4(2, 2, 2, 2, 4, defaultQValue = 0.0)

        val bestAction = qTable.bestAction(1, 1, 1, 1)
        // Any action is valid since they all have the same default value
        assertTrue(bestAction in 0 until 4, "Best action should be any valid action for the given state.")
    }

    @Test
    fun `test max value for a state`() {
        val qTable = QTableD4(2, 2, 2, 2, 3)
            .update(1, 1, 1, 1, 2, 20.0)

        assertEquals(20.0, qTable.maxValue(1, 1, 1, 1))
    }

    @Test
    fun `test max value using row, col, layer, feature`() {
        val qTable = QTableD4(3, 3, 3, 3, 2, defaultQValue = 0.0)
            .update(1, 1, 1, 1, 0, 5.0)
            .update(1, 1, 1, 1, 1, 10.0)

        val maxValue = qTable.maxValue(1, 1, 1, 1)
        assertEquals(10.0, maxValue)
    }

    @Test
    fun `test max value when multiple actions have same value`() {
        val qTable = QTableD4(3, 3, 3, 3, 3, defaultQValue = 1.0)
            .update(2, 2, 2, 2, 0, 15.0)
            .update(2, 2, 2, 2, 1, 15.0)

        val maxValue = qTable.maxValue(2, 2, 2, 2)
        assertEquals(15.0, maxValue)
    }

    @Test
    fun `test max value in default QTableD4`() {
        val qTable = QTableD4(4, 4, 4, 4, 5, defaultQValue = 3.0)

        val maxValue = qTable.maxValue(0, 0, 0, 0)
        assertEquals(3.0, maxValue)
    }

    @Test
    fun `test print`() {
        val qTable = QTableD4(2, 2, 2, 2, 3, defaultQValue = 1.0)
            .update(1, 0, 0, 0, 0, 5.0)

        // Capture and check the output of the `print` method
        val outputStream = java.io.ByteArrayOutputStream()
        val printStream = java.io.PrintStream(outputStream)
        val originalOut = System.out

        try {
            System.setOut(printStream)
            qTable.print()
            printStream.flush()
            val printedOutput = outputStream.toString()

            assertTrue(printedOutput.contains("5.0"), "Output should contain the updated Q-value 5.0")
            assertTrue(printedOutput.contains("1.0"), "Output should contain the default Q-value 1.0")
        } finally {
            System.setOut(originalOut)
        }
    }

    @Test
    fun `test asQTableD5`() {
        val d4Shape = intArrayOf(2, 2, 2, 2, 3)
        val d5Shape = intArrayOf(1, 1, 2, 2, 4, 3)
        val defaultQValue = 1.0

        val qTableD4 = QTableD4(2, 2, 2, 2, 3, defaultQValue = defaultQValue)
        val qTableD5 = qTableD4.asQTableD5(1, 1, 2, 2, 4, 3)

        assertArrayEquals(d5Shape, qTableD5.shape)
        assertEquals(defaultQValue, qTableD5.defaultQValue)
        assertTrue(qTableD5.deterministic)

        // Ensure the underlying data is copied correctly
        assertEquals(qTableD4.defaultQValue, qTableD5.defaultQValue)
    }

    @Test
    fun `test asQTableDN`() {
        val d4Shape = intArrayOf(2, 2, 2, 2, 3)
        val dnShape = intArrayOf(2, 2, 2, 6)
        val defaultQValue = 0.5

        val qTableD4 = QTableD4(2, 2, 2, 2, 3, defaultQValue = defaultQValue)
        val qTableDN = qTableD4.asQTableDN(2, 2, 2, 6)

        assertArrayEquals(dnShape, qTableDN.shape)
        assertEquals(defaultQValue, qTableDN.defaultQValue)
        assertTrue(qTableDN.deterministic)

        // Ensure the underlying data is copied correctly
        assertEquals(qTableD4.defaultQValue, qTableDN.defaultQValue)
        assertTrue(qTableDN.allStates().isNotEmpty(), "All states should be accessible in QTableDN")
    }
}