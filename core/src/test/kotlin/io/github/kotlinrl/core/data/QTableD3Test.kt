package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.*
import kotlin.test.*

class QTableD3Test {

    @Test
    fun `test save with default Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)
        val testFilePath = "test_qtable_default_qvalues.json"

        qTable.save(testFilePath)

        assertTrue(java.io.File(testFilePath).exists())
        java.io.File(testFilePath).delete() // Clean up
    }

    @Test
    fun `test save with updated Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(1, 1, 1, 1, 10.0)
        val testFilePath = "test_qtable_updated_qvalues.json"

        qTable.save(testFilePath)

        assertTrue(java.io.File(testFilePath).exists())
        java.io.File(testFilePath).delete() // Clean up
    }

    @Test
    fun `test toV with default Q-values`() {
        val qTable = QTableD3(2, 3, 3, 4, defaultQValue = 1.0)
        val vTable = qTable.toV()

        for (state in qTable.allStates()) {
            val expectedValue = 1.0 // All Q-values are initialized to the default value of 1.0
            assertEquals(expectedValue, vTable[state])
        }
    }

    @Test
    fun `test toV with updated Q-values`() {
        val qTable = QTableD3(2, 2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(0, 0, 0, 1, 3.0)
            .update(1, 1, 1, 0, 3.0)
            .update(1, 1, 1, 1, 7.0)

        val vTable = qTable.toV()

        assertEquals(5.0, vTable[mk.ndarray(mk[mk[0, 0, 0]])]) // Max of [5.0, 3.0, 0.0]
        assertEquals(7.0, vTable[mk.ndarray(mk[mk[1, 1, 1]])]) // Max of [0.0, 0.0, 7.0]
        assertEquals(0.0, vTable[mk.ndarray(mk[mk[0, 1, 0]])]) // Default Q-value
    }

    @Test
    fun `test toV with negative Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = -1.0)
            .update(0, 0, 0, 0, -3.0)
            .update(0, 0, 0, 1, -2.0)
            .update(1, 1, 1, 0, -5.0)
            .update(1, 1, 1, 1, -4.0)

        val vTable = qTable.toV()

        assertEquals(-2.0, vTable[mk.ndarray(mk[mk[0, 0, 0]])]) // Max of [-3.0, -2.0]
        assertEquals(-4.0, vTable[mk.ndarray(mk[mk[1, 1, 1]])]) // Max of [-5.0, -4.0]
    }

    @Test
    fun `test toV with custom shape`() {
        val qTable = QTableD3(3, 3, 3, 2, defaultQValue = 0.5)
            .update(2, 2, 2, 0, 8.0)
            .update(2, 2, 2, 1, 6.0)

        val vTable = qTable.toV()

        assertEquals(8.0, vTable[mk.ndarray(mk[mk[2, 2, 2]])]) // Max of [8.0, 6.0]
        assertEquals(0.5, vTable[mk.ndarray(mk[mk[0, 0, 0]])]) // Default Q-value
    }

    @Test
    fun `test bestAction with default Q-values`() {
        val qTable = QTableD3(2, 3, 3, 4, defaultQValue = 1.0)

        for (state in qTable.allStates()) {
            val expectedAction =
                0 // Since all Q-values are the same (1.0), the first action (index 0) should be returned
            assertEquals(expectedAction, qTable.bestAction(state))
        }
    }

    @Test
    fun `test bestAction with updated Q-values`() {
        val qTable = QTableD3(2, 2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(0, 0, 0, 1, 3.0)
            .update(1, 1, 1, 0, 3.0)
            .update(1, 1, 1, 1, 7.0)

        assertEquals(0, qTable.bestAction(0, 0, 0)) // Action 0 has max value 5.0
        assertEquals(1, qTable.bestAction(1, 1, 1)) // Action 1 has max value 7.0
    }

    @Test
    fun `test bestAction with negative Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = -1.0)
            .update(0, 0, 0, 0, -3.0)
            .update(0, 0, 0, 1, -2.0)
            .update(1, 1, 1, 0, -5.0)
            .update(1, 1, 1, 1, -4.0)

        assertEquals(1, qTable.bestAction(0, 0, 0)) // Action 1 has max value -2.0
        assertEquals(1, qTable.bestAction(1, 1, 1)) // Action 1 has max value -4.0
    }

    @Test
    fun `test load with default Q-values`() {
        val originalQTable = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)
        val testFilePath = "test_qtable_default_qvalues_load.json"

        originalQTable.save(testFilePath)
        val loadedQTable = QTableD3(2, 2, 2, 2)
        loadedQTable.load(testFilePath)

        for (state in originalQTable.allStates()) {
            for (action in 0 until 2) {
                assertEquals(originalQTable[state, action], loadedQTable[state, action])
            }
        }

        java.io.File(testFilePath).delete() // Clean up
    }

    @Test
    fun `test load with updated Q-values`() {
        val originalQTable = QTableD3(2, 2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(1, 1, 1, 1, 10.0)
        val testFilePath = "test_qtable_updated_qvalues_load.json"

        originalQTable.save(testFilePath)
        val loadedQTable = QTableD3(2, 2, 2, 2)
        loadedQTable.load(testFilePath)

        for (state in originalQTable.allStates()) {
            for (action in 0 until 2) {
                assertEquals(originalQTable[state, action], loadedQTable[state, action])
            }
        }

        java.io.File(testFilePath).delete() // Clean up
    }

    @Test
    fun `test print with default Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)

        // Capture the output of the print function
        val outputStream = java.io.ByteArrayOutputStream()
        System.setOut(java.io.PrintStream(outputStream))

        qTable.print()

        val output = outputStream.toString().trim()
        assertTrue(output.contains("1.0"), "Output should contain the default Q-value 1.0")

        // Restore the original System.out
        System.setOut(System.out)
    }

    @Test
    fun `test print with updated Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(1, 1, 1, 1, 10.0)

        // Capture the output of the print function
        val outputStream = java.io.ByteArrayOutputStream()
        System.setOut(java.io.PrintStream(outputStream))

        qTable.print()

        val output = outputStream.toString().trim()
        assertTrue(output.contains("5.0"), "Output should contain the Q-value 5.0")
        assertTrue(output.contains("10.0"), "Output should contain the Q-value 10.0")

        // Restore the original System.out
        System.setOut(System.out)
    }

    @Test
    fun `test asQTableD4`() {
        val qTableD3 = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)
            .update(0, 0, 0, 0, 5.0)
            .update(0, 0, 0, 1, 3.0)
            .update(1, 0, 0, 0, 2.4)
            .update(1, 0, 1, 0, 4.4)

        val qTableD4 = qTableD3.asQTableD4(2, 2, 2, 1, 2)

        assertEquals(listOf(2, 2, 2, 1, 2), qTableD4.shape.toList())
        assertEquals(5.0, qTableD4[0, 0, 0, 0, 0])
        assertEquals(3.0, qTableD4[0, 0, 0, 0, 1])
        assertEquals(2.4, qTableD4[1, 0, 0, 0, 0])
        assertEquals(4.4, qTableD4[1, 0, 1, 0, 0])
    }

    @Test
    fun `test asQTableD5`() {
        val qTableD3 = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)
            .update(0, 0, 0, 0, 5.0)
            .update(0, 0, 0, 1, 3.0)
            .update(1, 0, 0, 0, 2.4)
            .update(1, 0, 1, 0, 4.4)

        val qTableD5 = qTableD3.asQTableD5(2, 2, 2, 1, 1, 2)

        assertEquals(listOf(2, 2, 2, 1, 1, 2), qTableD5.shape.toList())
        assertEquals(5.0, qTableD5[0, 0, 0, 0, 0, 0])
        assertEquals(3.0, qTableD5[0, 0, 0, 0, 0, 1])
        assertEquals(2.4, qTableD5[1, 0, 0, 0, 0, 0])
        assertEquals(4.4, qTableD5[1, 0, 1, 0, 0, 0])
    }

    @Test
    fun `test asQTableD5 with updated Q-values`() {
        val qTableD3 = QTableD3(2, 2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(1, 1, 1, 1, 10.0)
        val qTableD5 = qTableD3.asQTableD5(2, 2, 2, 1, 1, 2)

        assertEquals(listOf(2, 2, 2, 1, 1, 2), qTableD5.shape.toList())
        assertEquals(5.0, qTableD5[0, 0, 0, 0, 0, 0])
        assertEquals(10.0, qTableD5[1, 1, 1, 0, 0, 1])
    }

    @Test
    fun `test asQTableD4 with updated Q-values`() {
        val qTableD3 = QTableD3(2, 2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(1, 1, 1, 1, 10.0)
        val qTableD4 = qTableD3.asQTableD4(2, 2, 2, 1, 2)

        assertEquals(listOf(2, 2, 2, 1, 2), qTableD4.shape.toList())
        assertEquals(5.0, qTableD4[0, 0, 0, 0, 0])
        assertEquals(10.0, qTableD4[1, 1, 1, 0, 1])
    }

    @Test
    fun `test asQTableDN`() {
        val qTableD3 = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)
            .update(0, 0, 0, 0, 5.0)
            .update(0, 0, 0, 1, 3.0)
            .update(1, 0, 0, 0, 2.4)
            .update(1, 0, 1, 0, 4.4)

        val qTableDN = qTableD3.asQTableDN(2, 2, 2, 1, 2)

        assertEquals(listOf(2, 2, 2, 1, 2), qTableDN.shape.toList())
        assertEquals(5.0, qTableDN[0, 0, 0, 0, 0])
        assertEquals(3.0, qTableDN[0, 0, 0, 0, 1])
        assertEquals(2.4, qTableDN[1, 0, 0, 0, 0])
        assertEquals(4.4, qTableDN[1, 0, 1, 0, 0])
    }

    @Test
    fun `test asQTableDN with updated Q-values`() {
        val qTableD3 = QTableD3(2, 2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(1, 1, 1, 1, 10.0)
        val qTableDN = qTableD3.asQTableDN(2, 2, 2, 1, 2)

        assertEquals(listOf(2, 2, 2, 1, 2), qTableDN.shape.toList())
        assertEquals(5.0, qTableDN[0, 0, 0, 0, 0])
        assertEquals(10.0, qTableDN[1, 1, 1, 0, 1])
    }

    @Test
    fun `test maxValue with default Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = 1.0)

        for (state in qTable.allStates()) {
            assertEquals(1.0, qTable.maxValue(state))
        }
    }

    @Test
    fun `test maxValue with updated Q-values`() {
        val qTable = QTableD3(2, 2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 0, 5.0)
            .update(0, 0, 0, 1, 3.0)
            .update(1, 1, 1, 0, 3.0)
            .update(1, 1, 1, 1, 7.0)

        assertEquals(5.0, qTable.maxValue(0, 0, 0)) // Max value in state (0, 0, 0)
        assertEquals(7.0, qTable.maxValue(1, 1, 1)) // Max value in state (1, 1, 1)
    }

    @Test
    fun `test maxValue with negative Q-values`() {
        val qTable = QTableD3(2, 2, 2, 2, defaultQValue = -1.0)
            .update(0, 0, 0, 0, -3.0)
            .update(0, 0, 0, 1, -2.0)
            .update(1, 1, 1, 0, -5.0)
            .update(1, 1, 1, 1, -4.0)

        assertEquals(-2.0, qTable.maxValue(0, 0, 0)) // Max value in state (0, 0, 0)
        assertEquals(-4.0, qTable.maxValue(1, 1, 1)) // Max value in state (1, 1, 1)
    }
}