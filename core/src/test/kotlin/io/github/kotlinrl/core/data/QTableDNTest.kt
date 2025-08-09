package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.*
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*

class QTableDNTest {

    /**
     * QTableDN class tests for the method toV.
     * This method maps the Q-value table to a V-value table by taking the
     * maximum value over actions for each state.
     */

    @Test
    fun `test toV with default Q values`() {
        val qTable = QTableDN(3, 3, deterministic = true, defaultQValue = 0.0)
        val vTable = qTable.toV()

        val expectedVTable = VTableDN(3)
        for (i in 0 until 3) {
            assertEquals(0.0, vTable[mk.ndarray(mk[i]).asDNArray()])
        }
    }

    @Test
    fun `test toV with custom Q values`() {
        val qTable = QTableDN(2, 2, deterministic = true, defaultQValue = 1.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 0, 2.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 1, 3.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 0, 4.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 1, 1.5)

        val vTable = qTable.toV()

        assertEquals(3.0, vTable[mk.ndarray(mk[0]).asDNArray()])
        assertEquals(4.0, vTable[mk.ndarray(mk[1]).asDNArray()])
    }

    @Test
    fun `test toV with deterministic false`() {
        val qTable = QTableDN(2, 3, deterministic = false, defaultQValue = 1.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 0, -1.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 1, 0.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 2, 2.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 0, 1.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 1, 5.0)

        val vTable = qTable.toV()

        assertEquals(2.0, vTable[mk.ndarray(mk[0]).asDNArray()])
        assertEquals(5.0, vTable[mk.ndarray(mk[1]).asDNArray()])
    }

    @Test
    fun `test toV with larger table`() {
        val qTable = QTableDN(2, 2, 2, deterministic = true, defaultQValue = 0.0)
            .update(mk.ndarray(mk[0, 0]).asDNArray(), 0, 2.0)
            .update(mk.ndarray(mk[0, 0]).asDNArray(), 1, 3.5)
            .update(mk.ndarray(mk[1, 1]).asDNArray(), 0, 4.2)
            .update(mk.ndarray(mk[1, 1]).asDNArray(), 1, 4.1)

        val vTable = qTable.toV()

        assertEquals(3.5, vTable[mk.ndarray(mk[0, 0]).asDNArray()])
        assertEquals(4.2, vTable[mk.ndarray(mk[1, 1]).asDNArray()])
    }

    @Test
    fun `test toV with negative Q values`() {
        val qTable = QTableDN(2, 2, deterministic = true, defaultQValue = -5.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 0, -3.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 1, -1.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 0, -4.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 1, -2.5)

        val vTable = qTable.toV()

        assertEquals(-1.0, vTable[mk.ndarray(mk[0]).asDNArray()])
        assertEquals(-2.5, vTable[mk.ndarray(mk[1]).asDNArray()])
    }

    @Test
    fun `test maxValue with default Q values`() {
        val qTable = QTableDN(2, 2, deterministic = true, defaultQValue = 0.0)
        for (state in qTable.allStates()) {
            assertEquals(0.0, qTable.maxValue(state))
        }
    }

    @Test
    fun `test maxValue with custom values`() {
        val qTable = QTableDN(2, 2, deterministic = true, defaultQValue = 0.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 0, 1.5)
            .update(mk.ndarray(mk[0]).asDNArray(), 1, 2.5)
            .update(mk.ndarray(mk[1]).asDNArray(), 0, 3.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 1, 4.0)

        assertEquals(2.5, qTable.maxValue(mk.ndarray(mk[0]).asDNArray()))
        assertEquals(4.0, qTable.maxValue(mk.ndarray(mk[1]).asDNArray()))
    }

    @Test
    fun `test maxValue with negative Q values`() {
        val qTable = QTableDN(2, 2, deterministic = true, defaultQValue = -5.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 0, -2.5)
            .update(mk.ndarray(mk[0]).asDNArray(), 1, -1.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 0, -3.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 1, -4.5)

        assertEquals(-1.0, qTable.maxValue(mk.ndarray(mk[0]).asDNArray()))
        assertEquals(-3.0, qTable.maxValue(mk.ndarray(mk[1]).asDNArray()))
    }

    @Test
    fun `test maxValue with mixed Q values`() {
        val qTable = QTableDN(2, 2, deterministic = true, defaultQValue = 0.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 0, -1.0)
            .update(mk.ndarray(mk[0]).asDNArray(), 1, 2.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 0, -3.0)
            .update(mk.ndarray(mk[1]).asDNArray(), 1, 1.0)

        assertEquals(2.0, qTable.maxValue(mk.ndarray(mk[0]).asDNArray()))
        assertEquals(1.0, qTable.maxValue(mk.ndarray(mk[1]).asDNArray()))
    }
}