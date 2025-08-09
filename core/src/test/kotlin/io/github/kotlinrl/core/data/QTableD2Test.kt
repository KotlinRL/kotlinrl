package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.io.*
import kotlin.test.*

/**
 * Test class for QTableD2's toV function.
 *
 * QTableD2 represents a 3D Q-table that maps states (represented as 2D arrays)
 * and actions to Q-values. The toV function converts the Q-table into a V-table
 * (a 2D table) by calculating the maximum Q-value for each state across all
 * possible actions.
 */
class QTableD2Test {

    @Test
    fun `bestAction should return correct action for highest Q-value`() {
        val qTable = QTableD2(3, 3, 3, defaultQValue = 0.0)
            .update(1, 1, 0, 2.0)
            .update(1, 1, 1, 5.0)
            .update(1, 1, 2, 3.0)

        assertEquals(1, qTable.bestAction(1, 1))
    }

    @Test
    fun `bestAction should return first action when all Q-values are equal`() {
        val qTable = QTableD2(2, 2, 3, defaultQValue = 1.0)

        assertEquals(0, qTable.bestAction(0, 0))
        assertEquals(0, qTable.bestAction(1, 1))
    }

    @Test
    fun `bestAction should handle negative Q-values correctly`() {
        val qTable = QTableD2(2, 2, 3, defaultQValue = -2.0)
            .update(0, 0, 0, -1.0)
            .update(0, 0, 1, -3.0)
            .update(0, 0, 2, -4.0)

        assertEquals(0, qTable.bestAction(0, 0))
    }

    @Test
    fun `bestAction should return consistent result with deterministic flag`() {
        val qTable = QTableD2(2, 2, 2, defaultQValue = 0.0, deterministic = true)
            .update(1, 1, 0, 5.0)
            .update(1, 1, 1, 5.0)

        assertEquals(0, qTable.bestAction(1, 1))
    }

    @Test
    fun `get should return correct value at specified state and action`() {
        val qTable = QTableD2(3, 3, 3, defaultQValue = 0.0)
            .update(1, 2, 0, 5.0)
            .update(1, 2, 1, 10.0)

        assertEquals(5.0, qTable.get(1, 2, 0))
        assertEquals(10.0, qTable.get(1, 2, 1))
        assertEquals(0.0, qTable.get(1, 2, 2))

        assertEquals(5.0, qTable.get(1, 2, 0))
        assertEquals(10.0, qTable.get(1, 2, 1))
        assertEquals(0.0, qTable.get(1, 2, 2))
    }

    @Test
    fun `get should return default value for uninitialized states`() {
        val qTable = QTableD2(2, 2, 2, defaultQValue = -1.0)

        assertEquals(-1.0, qTable.get(0, 0, 0))
        assertEquals(-1.0, qTable.get(0, 1, 1))
        assertEquals(-1.0, qTable.get(1, 0, 0))
        assertEquals(-1.0, qTable.get(1, 1, 1))

        assertEquals(-1.0, qTable.get(0, 0, 0))
        assertEquals(-1.0, qTable.get(0, 1, 1))
        assertEquals(-1.0, qTable.get(1, 0, 0))
        assertEquals(-1.0, qTable.get(1, 1, 1))
    }

    @Test
    fun `toV should return a VTableD2 with correct maximum values for a simple QTableD2`() {
        var qTable = QTableD2(2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 1.5)
            .update(0, 0, 1, 2.5)
            .update(0, 1, 0, 3.0)
            .update(0, 1, 1, 1.0)
            .update(1, 0, 0, 0.0)
            .update(1, 0, 1, 5.0)
            .update(1, 1, 0, 4.0)
            .update(1, 1, 1, 3.5)

        val vTable = qTable.toV()

        assertEquals(2.5, vTable[0, 0])
        assertEquals(3.0, vTable[0, 1])
        assertEquals(5.0, vTable[1, 0])
        assertEquals(4.0, vTable[1, 1])
    }

    @Test
    fun `toV should handle default Q-values correctly`() {
        val qTable = QTableD2(3, 3, 2, defaultQValue = 1.0)

        val vTable = qTable.toV()

        for (state in qTable.allStates()) {
            assertEquals(1.0, vTable[state])
        }
    }

    @Test
    fun `toV should handle negative Q-values correctly`() {
        val qTable = QTableD2(2, 2, 3, defaultQValue = -1.0)
            .update(0, 0, 0, -2.0)
            .update(0, 0, 1, -3.0)
            .update(0, 0, 2, -0.5)
            .update(1, 1, 0, -4.0)
            .update(1, 1, 1, -1.5)
            .update(1, 1, 2, -2.0)

        val vTable = qTable.toV()

        assertEquals(-0.5, vTable[0, 0])
        assertEquals(-1.0, vTable[0, 1])
        assertEquals(-1.0, vTable[1, 0])
        assertEquals(-1.5, vTable[1, 1])
    }

    @Test
    fun `toV should work correctly for a larger QTableD2`() {
        val qTable = QTableD2(5, 5, 4, defaultQValue = 0.0)
            .update(2, 3, 0, 10.0)
            .update(2, 3, 1, 15.0)
            .update(2, 3, 2, 12.0)
            .update(2, 3, 3, 9.0)
            .update(4, 4, 0, 20.0)

        val vTable = qTable.toV()

        assertEquals(15.0, vTable[2, 3])
        assertEquals(20.0, vTable[4, 4])
        for (state in qTable.allStates().filterNot {
            it.toIntArray().contentEquals(intArrayOf(2, 3)) || it.toIntArray().contentEquals(intArrayOf(4, 4))
        }) {
            assertEquals(0.0, vTable[state])
        }
    }

    @Test
    fun `toV should maintain independent copies of QTableD2 and VTableD2`() {
        var qTable = QTableD2(2, 2, 2, defaultQValue = 0.0)
            .update(0, 0, 0, 1.2)
            .update(0, 0, 1, 3.4)

        val vTable = qTable.toV()

        qTable = qTable.update(0, 0, 1, 5.0)

        assertEquals(3.4, vTable[0, 0]) // The VTableD2 should not be affected
        assertEquals(5.0, qTable.maxValue(0, 0))
    }

    @Test
    fun `save should save QTableD2 to a file and allow reloading it with the correct data`() {
        val tempFilePath = createTempFile("qtable_test", ".bin").absolutePath
        try {
            val qTable = QTableD2(3, 3, 3, defaultQValue = 0.0)
                .update(1, 1, 0, 2.0)
                .update(1, 1, 2, 6.0)

            qTable.save(tempFilePath)

            val loadedQTable = QTableD2(3, 3, 3, defaultQValue = 0.0)
            loadedQTable.load(tempFilePath)

            assertEquals(2.0, loadedQTable.get(1, 1, 0))
            assertEquals(6.0, loadedQTable.get(1, 1, 2))
            assertEquals(0.0, loadedQTable.get(1, 1, 1))
        } finally {
            File(tempFilePath).delete()
        }
    }

    @Test
    fun `print should output QTableD2 content correctly`() {
        val qTable = QTableD2(2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 1.5)
            .update(0, 1, 2, 2.5)

        val outputStream = java.io.ByteArrayOutputStream()
        val printStream = java.io.PrintStream(outputStream)
        val originalOut = System.out

        try {
            System.setOut(printStream)
            qTable.print()
            System.out.flush()

            val output = outputStream.toString()
            assertTrue(output.isNotEmpty())
            assertEquals(
                """
                [[[1.5, 0.0, 0.0],
                [0.0, 0.0, 2.5]],

                [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]]
                
            """.trimIndent(), output
            )
        } finally {
            System.setOut(originalOut)
        }
    }

    @Test
    fun `asQTableD3 should convert QTableD2 to QTableD3 correctly`() {
        val qTableD2 = QTableD2(2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 1.2)
            .update(0, 0, 1, 2.5)
            .update(1, 1, 2, 3.4)

        val qTableD3 = qTableD2.asQTableD3(2, 2, 1, 3)

        // Check if values are correctly copied
        assertEquals(1.2, qTableD3[0, 0, 0, 0])
        assertEquals(2.5, qTableD3[0, 0, 0, 1])
        assertEquals(3.4, qTableD3[1, 1, 0, 2])
        assertEquals(0.0, qTableD3[1, 1, 0, 0]) // Default value

        // Verify updates don't affect the original QTableD2
        qTableD3.update(0, 0, 0, 0, 5.0)
        assertEquals(1.2, qTableD2[0, 0, 0]) // Original should remain unchanged
    }

    @Test
    fun `asQTableD4 should convert QTableD2 to QTableD4 correctly`() {
        val qTableD2 = QTableD2(2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 1.2)
            .update(0, 0, 1, 2.5)
            .update(1, 1, 2, 3.4)

        val qTableD4 = qTableD2.asQTableD4(2, 2, 1, 1, 3)

        // Check if values are correctly copied
        assertEquals(1.2, qTableD4[0, 0, 0, 0, 0])
        assertEquals(2.5, qTableD4[0, 0, 0, 0, 1])
        assertEquals(3.4, qTableD4[1, 1, 0, 0, 2])
        assertEquals(0.0, qTableD4[1, 1, 0, 0, 0]) // Default value

        // Verify updates don't affect the original QTableD2
        qTableD4.update(0, 0, 0, 0, 0, 5.0)
        assertEquals(1.2, qTableD2[0, 0, 0]) // Original should remain unchanged
    }

    @Test
    fun `asQTableD5 should convert QTableD2 to QTableD5 correctly`() {
        val qTableD2 = QTableD2(2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 1.3)
            .update(0, 0, 1, 2.7)
            .update(1, 1, 2, 4.0)

        val qTableD5 = qTableD2.asQTableD5(2, 2, 1, 1, 1, 3)

        // Check that values are correctly copied to QTableD5
        assertEquals(1.3, qTableD5[0, 0, 0, 0, 0, 0])
        assertEquals(2.7, qTableD5[0, 0, 0, 0, 0, 1])
        assertEquals(4.0, qTableD5[1, 1, 0, 0, 0, 2])
        assertEquals(0.0, qTableD5[1, 1, 0, 0, 0, 0]) // Default value

        // Confirm independence of QTableD5 updates
        qTableD5.update(0, 0, 0, 0, 0, 0, 6.0)
        assertEquals(1.3, qTableD2[0, 0, 0 ])// Original should remain unchanged
    }

    @Test
    fun `update should correctly update the Q-value for a specific state-action pair`() {
        val qTable = QTableD2(3, 3, 3, defaultQValue = 0.0)
        val updatedTable = qTable.update(0, 0, 1, 4.5)

        assertEquals(4.5, updatedTable[0, 0, 1])
        assertEquals(0.0, updatedTable[0, 0, 0]) // Ensure other values remain default
    }

    @Test
    fun `update should correctly update multiple Q-values`() {
        val qTable = QTableD2(3, 3, 3, defaultQValue = 0.0)
        val updatedTable = qTable
            .update(1, 1, 0, 2.0)
            .update(1, 1, 1, 3.0)

        assertEquals(2.0, updatedTable[1, 1, 0])
        assertEquals(3.0, updatedTable[1, 1, 1])
        assertEquals(0.0, updatedTable[1, 1, 2]) // Check default
    }

    @Test
    fun `update should not modify existing QTableD2 instance`() {
        val qTable = QTableD2(2, 2, 2, defaultQValue = 0.0)
        val updatedTable = qTable.update(0, 0, 1, 3.5)

        assertEquals(0.0, qTable[0, 0, 1]) // Original should remain unchanged
        assertEquals(3.5, updatedTable[0, 0, 1]) // Updated instance reflects changes
    }

    @Test
    fun `update using row, col, action method should correctly update Q-value`() {
        val qTable = QTableD2(2, 2, 2, defaultQValue = 0.0)
        val updatedTable = qTable.update(0, 1, 1, 5.0)

        assertEquals(5.0, updatedTable[0, 1, 1])
        assertEquals(0.0, updatedTable[0, 1, 0]) // Ensure other values remain default
    }

    @Test
    fun `asQTableDN should correctly convert QTableD2 to QTableDN`() {
        val qTableD2 = QTableD2(2, 2, 3, defaultQValue = 0.0)
            .update(0, 0, 0, 1.3)
            .update(0, 0, 1, 2.7)
            .update(1, 1, 2, 4.0)

        val qTableDN = qTableD2.asQTableDN(2, 2, 1, 3)

        // Check that values are correctly copied to QTableDN
        assertEquals(1.3, qTableDN[0, 0, 0, 0])
        assertEquals(2.7, qTableDN[0, 0, 0, 1])
        assertEquals(4.0, qTableDN[1, 1, 0, 2])
        assertEquals(0.0, qTableDN[1, 1, 0, 0]) // Default value

        // Confirm independence of QTableDN updates
        qTableDN.update(mk.ndarray<Int>(mk[mk[mk[0, 0, 0]]]).asDNArray(), 0, 6.0)
        assertEquals(1.3, qTableD2[0, 0, 0]) // Original should remain unchanged
    }
}