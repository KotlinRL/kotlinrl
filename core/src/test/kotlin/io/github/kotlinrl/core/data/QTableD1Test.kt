package io.github.kotlinrl.core.data

import java.io.*
import kotlin.io.path.*
import kotlin.test.*

class QTableD1Test {

    @Test
    fun `test get retrieves correct default Q-value`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.5)

        // Act
        val qValue = qTable[2, 3]

        // Assert
        assertEquals(0.5, qValue)
    }

    @Test
    fun `test update modifies Q-value correctly`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.5)

        // Act
        val updatedQTable = qTable.update(2, 3, 1.5)
        val updatedValue = updatedQTable[2, 3]

        // Assert
        assertEquals(1.5, updatedValue)
    }

    @Test
    fun `test allStates returns all possible states`() {
        // Arrange
        val qTable = QTableD1(10, 5)

        // Act
        val states = qTable.allStates()

        // Assert
        assertEquals((0..9).toList(), states)
    }

    @Test
    fun `test maxValue returns correct maximum Q-value`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.0)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(0, 2, 3.0)

        // Act
        val maxValue = qTable.maxValue(0)

        // Assert
        assertEquals(3.0, maxValue)
    }

    @Test
    fun `test bestAction returns correct action`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.0)
            .update(0, 0, 1.0)
            .update(0, 1, 2.0)
            .update(0, 2, 3.0)

        // Act
        val bestAction = qTable.bestAction(0)

        // Assert
        assertEquals(2, bestAction)
    }

    @Test
    fun `test copy creates an independent QTableD1`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.0)
        val qTableCopy = qTable.copy()

        // Act
        val modifiedQTable = qTableCopy.update(0, 0, 5.0)

        // Assert
        assertEquals(5.0, modifiedQTable.get(0, 0))
        // Original QTableD1 should remain unchanged
        assertEquals(0.0, qTable.get(0, 0))
    }

    @Test
    fun `test load function correctly restores QTableD1 from file system`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.5)
            .update(2, 3, 1.5)
        val tempFile = createTempFile()

        // Act
        qTable.save(tempFile.toString())
        val loadedQTable = QTableD1(10, 5).apply { load(tempFile.toString()) }

        // Assert
        assertEquals(1.5, loadedQTable[2, 3])
        assertEquals(0.5, loadedQTable[0, 0])

        // Cleanup
        tempFile.toFile().delete()
    }

    @Test
    fun `test load function throws exception on invalid file`() {
        // Arrange
        val invalidPath = "/invalid_path/qtable_file"
        val qTable = QTableD1(10, 5)

        // Act & Assert
        val exception = assertFailsWith<Exception> {
            qTable.load(invalidPath)
        }
        assertNotNull(exception.message)
    }

    @Test
    fun `test copy(true) creates a deterministic QTableD1`() {
        val qTable: QTableD1 = QTableD1(5, 3, deterministic = false, defaultQValue = 0.0)
        assertFalse(qTable.deterministic)
        val deterministicQTable = qTable.copy(true)
        assertTrue(deterministicQTable.deterministic)
    }

    @Test
    fun `test toV converts to ValueFunction with correct max Q-values when stochastic`() {
        // Arrange
        val qTable = QTableD1(5, 3, deterministic = false, defaultQValue = 0.0)
            .update(0, 0, 1.0)
            .update(1, 1, 2.0)
            .update(2, 2, 3.0)


        // Act
        val V = qTable.toV()

        // Assert
        assertEquals(1.0, V[0])
        assertEquals(2.0, V[1])
        assertEquals(3.0, V[2])
        // Default Q-value ensures the remaining default values are 0.0
        assertEquals(0.0, V[3])
        assertEquals(0.0, V[4])
    }

    @Test
    fun `test toV converts to ValueFunction with correct max Q-values when deterministic`() {
        // Arrange
        val qTable = QTableD1(5, 3, deterministic = true, defaultQValue = 0.0)
            .update(0, 0, 1.0)
            .update(1, 1, 2.0)
            .update(2, 2, 3.0)


        // Act
        val V = qTable.toV()

        // Assert
        assertEquals(1.0, V[0])
        assertEquals(2.0, V[1])
        assertEquals(3.0, V[2])
        // Default Q-value ensures the remaining default values are 0.0
        assertEquals(0.0, V[3])
        assertEquals(0.0, V[4])
    }

    @Test
    fun `test save function writes QTableD1 to file system`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.5)
        val tempFile = createTempFile()

        // Act
        qTable.save(tempFile.toString())

        // Assert
        assertTrue(tempFile.toFile().exists())

        // Cleanup
        tempFile.toFile().delete()
    }

    @Test
    fun `test save function throws exception on invalid path`() {
        // Arrange
        val qTable = QTableD1(10, 5, defaultQValue = 0.5)
        val invalidPath = "/invalid_path/qtable_file"

        // Act & Assert
        val exception = assertFailsWith<Exception> {
            qTable.save(invalidPath)
        }
        assertNotNull(exception.message)
    }

    @Test
    fun `test print outputs correct structure`() {
        // Arrange
        val qTable = QTableD1(3, 2, defaultQValue = 0.0)
            .update(0, 0, 1.5)
            .update(1, 1, 3.0)
        val outputStream = ByteArrayOutputStream()
        val printStream = PrintStream(outputStream)
        val originalOut = System.out
        System.setOut(printStream)

        try {
            // Act
            qTable.print()
            System.out.flush()
            val output = outputStream.toString().trim()

            // Assert
            // This checks the structure without verifying the exact content
            assertTrue(output.isNotEmpty(), "QTableD1 print output should not be empty")
            assertEquals( """
                    [[1.5, 0.0],
                    [0.0, 3.0],
                    [0.0, 0.0]]
                    """.trimIndent(), output)
        } finally {
            // Restore original System.out
            System.setOut(originalOut)
        }
    }

    @Test
    fun `test asQTableD2 creates a valid QTableD2 instance`() {
        // Arrange
        val qTableD1 = QTableD1(10, 5, deterministic = true, defaultQValue = 0.5)

        // Act
        val qTableD2 = qTableD1.asQTableD2(5, 5, 5)

        // Assert
        assertEquals(listOf(5, 5, 5), qTableD2.shape.toList())
        assertEquals(0.5, qTableD2.defaultQValue)
        assertTrue(qTableD2.deterministic)
    }

    @Test
    fun `test asQTableD3 creates a valid QTableD3 instance`() {
        // Arrange
        val qTableD1 = QTableD1(9, 5, deterministic = true, defaultQValue = 0.5)

        // Act
        val qTableD3 = qTableD1.asQTableD3(3, 3, 3, 5)

        // Assert
        assertEquals(listOf(3, 3, 3, 5), qTableD3.shape.toList())
        assertEquals(0.5, qTableD3.defaultQValue)
        assertTrue(qTableD3.deterministic)
    }

    @Test
    fun `test asQTableD3 preserves values and attributes`() {
        // Arrange
        val qTableD1 = QTableD1(9, 3, deterministic = false, defaultQValue = 1.0)
            .update(1, 2, 3.5)

        // Act
        val qTableD3 = qTableD1.asQTableD3(3, 3, 3, 3)

        // Assert
        assertEquals(3.5, qTableD3[0, 0, 1, 2])
        assertEquals(1.0, qTableD3[0, 0, 1, 0])
        assertFalse(qTableD3.deterministic)
    }

    @Test
    fun `test asQTableD4 creates a valid QTableD4 instance`() {
        // Arrange
        val qTableD1 = QTableD1(20, 5, deterministic = true, defaultQValue = 0.5)

        // Act
        val qTableD4 = qTableD1.asQTableD4(1, 1, 5, 5, 4)

        // Assert
        assertEquals(listOf(1, 1, 5, 5, 4), qTableD4.shape.toList())
        assertEquals(0.5, qTableD4.defaultQValue)
        assertTrue(qTableD4.deterministic)
    }

    @Test
    fun `test asQTableD4 preserves values and attributes`() {
        // Arrange
        val qTableD1 = QTableD1(20, 3, deterministic = false, defaultQValue = 1.0)
            .update(5, 2, 4.5)

        // Act
        val qTableD4 = qTableD1.asQTableD4(1, 1, 5, 5, 4)

        // Assert
        assertEquals(4.5, qTableD4[0, 0, 0, 4, 1])
        assertEquals(1.0, qTableD4[0, 0, 0, 0, 0])
        assertFalse(qTableD4.deterministic)
    }

    @Test
    fun `test asQTableD5 creates a valid QTableD5 instance`() {
        // Arrange
        val qTableD1 = QTableD1(30, 5, deterministic = true, defaultQValue = 0.5)

        // Act
        val qTableD5 = qTableD1.asQTableD5(2, 3, 5, 1, 1, 5)

        // Assert
        assertEquals(listOf(2, 3, 5, 1, 1, 5), qTableD5.shape.toList())
        assertEquals(0.5, qTableD5.defaultQValue)
        assertTrue(qTableD5.deterministic)
    }

    @Test
    fun `test asQTableD5 preserves values and attributes`() {
        // Arrange
        val qTableD1 = QTableD1(30, 5, deterministic = false, defaultQValue = 1.0)
            .update(15, 2, 4.5)

        // Act
        val qTableD5 = qTableD1.asQTableD5(2, 3, 5, 1, 1, 5)

        // Assert
        assertEquals(4.5, qTableD5[1, 0, 0, 0, 0, 2])
        assertEquals(1.0, qTableD5[0, 0, 0, 0, 0, 0])
        assertFalse(qTableD5.deterministic)
    }

    @Test
    fun `test asQTableDN creates a valid QTableDN instance`() {
        // Arrange
        val qTableD1 = QTableD1(30, 5, deterministic = true, defaultQValue = 0.5)

        // Act
        val qTableDN = qTableD1.asQTableDN(30, 5)

        // Assert
        assertEquals(listOf(30, 5), qTableDN.shape.toList())
        assertEquals(0.5, qTableDN.defaultQValue)
        assertTrue(qTableDN.deterministic)
    }

    @Test
    fun `test asQTableDN creates a valid multi-dimensional QTableDN instance`() {
        // Arrange
        val qTableD1 = QTableD1(30, 5, deterministic = true, defaultQValue = 0.5)

        // Act
        val qTableDN = qTableD1.asQTableDN(1, 1, 1, 2, 3, 5, 1, 1, 5)

        // Assert
        assertEquals(listOf(1, 1, 1, 2, 3, 5, 1, 1, 5), qTableDN.shape.toList())
        assertEquals(0.5, qTableDN.defaultQValue)
        assertTrue(qTableDN.deterministic)
    }

    @Test
    fun `test asQTableDN preserves values and attributes`() {
        // Arrange
        val qTableD1 = QTableD1(30, 5, deterministic = false, defaultQValue = 0.8)
            .update(10, 2, 3.7)

        // Act
        val qTableDN = qTableD1.asQTableDN(1, 1, 1, 2, 3, 5, 1, 1, 5)

        // Assert
        assertEquals(3.7, qTableDN[0, 0, 0, 0, 2, 0, 0, 0, 2])
        assertEquals(0.8, qTableDN[0, 0, 0, 0, 0, 0, 0, 0, 0])
        assertFalse(qTableDN.deterministic)
    }

    @Test
    fun `test asQTableDN throws exception with invalid shape`() {
        // Arrange
        val qTableD1 = QTableD1(10, 5)

        // Act & Assert
        val exception = assertFailsWith<IllegalArgumentException> {
            qTableD1.asQTableDN(10) // Invalid shape
        }
        assertEquals("QTableDN shape requires at least 2 arguments", exception.message)
    }
}
