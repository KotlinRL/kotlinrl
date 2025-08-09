package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray
import org.junit.jupiter.api.Assertions.assertDoesNotThrow
import java.io.FileNotFoundException
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class VTableD4Test {

    @Test
    fun `test max returns largest value`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(2, 3, 4, 5)

        // Update specific values in the table
        vTable = vTable.update(0, 1, 2, 3, 5.7)
            .update(1, 2, 3, 4, 10.2)

        // Assert max value is the largest updated value
        assertEquals(10.2, vTable.max(), "The max function should return the largest value in the table.")
    }

    @Test
    fun `test max on empty table returns zero`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Assert that max is zero for an unmodified table
        assertEquals(0.0, vTable.max(), "The max function should return 0 for an empty or unmodified table.")
    }

    @Test
    fun `test max with multiple updates`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(3, 3, 2, 4)

        // Update multiple values in the table
        vTable = vTable.update(0, 0, 0, 0, 2.0)
            .update(1, 1, 1, 1, 15.3)
            .update(2, 2, 1, 3, 9.8)

        // Assert max value is the largest among all updates
        assertEquals(
            15.3,
            vTable.max(),
            "The max function should correctly find the highest value after multiple updates."
        )
    }

    /**
     * Tests for the `asVTableN` function in the VTableD4 class.
     * The `asVTableN` function allows the conversion of a `VTableD4` object into
     * a `VTableDN` object with an arbitrary shape while maintaining the data content.
     */

    @Test
    fun `test asVTableN with valid shape resulting in similar size`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(2, 3, 4, 5)

        // Update a few values in the table
        vTable = vTable.update(0, 0, 0, 0, 1.0)
            .update(1, 2, 3, 4, 2.5)

        // Convert to a VTableDN with compatible dimensional size
        val vTableN = vTable.asVTableN(6, 4, 5)

        // Verify shape of resulting table
        assertContentEquals(intArrayOf(6, 4, 5), vTableN.shape)

        // Verify that data is preserved
        val expectedData = vTableN.table.toDoubleArray()
        val originalData = vTable.base.table.toDoubleArray()
        assertContentEquals(originalData, expectedData)
    }

    @Test
    fun `test asVTableN with valid reshaped dimensions`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(2, 3, 4, 5)

        // Update a value in the table
        vTable = vTable.update(1, 1, 1, 1, 7.5)

        // Convert to another VTableDN with different dimensions
        val vTableN = vTable.asVTableN(24, 5)

        // Verify shape of resulting table
        assertContentEquals(intArrayOf(24, 5), vTableN.shape)

        // Verify that data is preserved
        val expectedData = vTableN.table.toDoubleArray()
        val originalData = vTable.base.table.toDoubleArray()
        assertContentEquals(originalData, expectedData)
    }

    @Test
    fun `test asVTableN with single dimension shape`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(2, 3, 4, 5)

        // Update a value in the table
        vTable = vTable.update(0, 2, 3, 4, 3.14)

        // Convert to a VTableDN with a single-dimension shape
        val vTableN = vTable.asVTableN(120)

        // Verify shape of resulting table
        assertContentEquals(intArrayOf(120), vTableN.shape)

        // Verify that data is preserved
        val expectedData = vTableN.table.toDoubleArray()
        val originalData = vTable.base.table.toDoubleArray()
        assertContentEquals(originalData, expectedData)
    }

    @Test
    fun `test save writes data to valid path`() {
        // Create a temporary file for saving
        val tempFile = createTempFile("vtable_save", ".dat")

        try {
            // Initialize a VTableD4 instance and update some values
            val vTable = VTableD4(2, 3, 4, 5)
                .update(0, 0, 0, 0, 1.0)
                .update(1, 2, 3, 4, 2.5)

            // Save the table to the temporary file
            assertDoesNotThrow {
                vTable.save(tempFile.absolutePath)
            }

            // Check that the file is not empty
            assert(tempFile.exists() && tempFile.length() > 0)
        } finally {
            // Clean up the temporary file
            tempFile.delete()
        }
    }

    @Test
    fun `test save throws exception for invalid path`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Attempt to save to an invalid path and verify exception is thrown
        val exception = kotlin.test.assertFailsWith<FileNotFoundException> {
            vTable.save("/invalid/path/vtable_save.dat")
        }

        // Verify the exception contains a message
        assertNotNull(exception.message)
    }

    @Test
    fun `test asVTableN with invalid shape throws exception`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Expect an exception for incompatible shape
        val exception = kotlin.test.assertFailsWith<ArrayIndexOutOfBoundsException> {
            vTable.asVTableN(1, 10, 7)
        }

        // Verify exception message
        assertNotNull(exception.message)
    }

    @Test
    fun `test asVTable5 with valid dimensions`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(2, 3, 4, 5)

        // Update a few values in the table
        vTable = vTable.update(0, 0, 0, 0, 1.0)
            .update(1, 2, 3, 4, 2.5)

        // Convert to a VTableD5 with valid dimensions
        val vTable5 = vTable.asVTable5(2, 3, 4, 5, 1)

        // Verify shape of resulting table
        assertContentEquals(intArrayOf(2, 3, 4, 5, 1), vTable5.shape)
    }

    @Test
    fun `test asVTable5 preserves data`() {
        // Initialize a VTableD4 instance
        var vTable = VTableD4(2, 3, 4, 5)

        // Update specific values in the table
        vTable = vTable.update(0, 1, 2, 3, 4.2)
            .update(1, 0, 3, 4, 8.6)

        // Convert to a VTableD5 with valid dimensions
        val vTable5 = vTable.asVTable5(2, 3, 4, 5, 1)

        // Verify that data is preserved
        val expectedData = vTable5.base.table.toDoubleArray()
        val originalData = vTable.base.table.toDoubleArray()
        assertContentEquals(originalData, expectedData)
    }

    @Test
    fun `test asVTable5 with invalid dimensions throws exception`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Expect an exception for incompatible arguments
        val exception = kotlin.test.assertFailsWith<ArrayIndexOutOfBoundsException> {
            vTable.asVTable5(1, 6, 1, 1, 2)
        }

        // Verify exception message
        assertNotNull(exception.message)
    }

    @Test
    fun `test print executes without exception`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Update specific values in the table
        var updatedVTable = vTable.update(0, 1, 2, 3, 3.14)
        updatedVTable = updatedVTable.update(1, 2, 3, 4, 1.618)

        // Verify the print functionality does not cause any exceptions; capture output if needed
        assertDoesNotThrow {
            updatedVTable.print()
        }
    }

    @Test
    fun `test load with valid path`() {
        // Create a temporary file for saving and loading
        val tempFile = createTempFile("vtable", ".dat")

        try {
            // Initialize a VTableD4 instance and update values
            val vTable = VTableD4(2, 3, 4, 5)
                .update(0, 0, 0, 0, 1.0)
                .update(1, 2, 3, 4, 2.5)

            // Save the table to the temporary file
            vTable.save(tempFile.absolutePath)

            // Load a new VTableD4 instance from the saved file
            val loadedVTable = VTableD4(2, 3, 4, 5)
            loadedVTable.load(tempFile.absolutePath)

            // Verify the data in the loaded table matches the original
            assertContentEquals(
                vTable.base.table.toDoubleArray(),
                loadedVTable.base.table.toDoubleArray()
            )
        } finally {
            // Clean up the temporary file
            tempFile.delete()
        }
    }

    @Test
    fun `test load throws exception for invalid path`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Attempt to load from an invalid path and verify exception is thrown
        val exception = kotlin.test.assertFailsWith<IllegalArgumentException> {
            vTable.load("/invalid/path/vtable.dat")
        }

        // Verify exception message
        assertNotNull(exception.message)
    }

    @Test
    fun `test allStates generates correct number of states`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Calculate the expected number of states
        val expectedStateCount = 2 * 3 * 4 * 5

        // Verify that allStates returns the correct count
        assertEquals(expectedStateCount, vTable.allStates().size)
    }

    @Test
    fun `test allStates generates valid states`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Get all states
        val allStates = vTable.allStates()

        // Verify that each state matches the table's shape
        allStates.forEach { state ->
            val stateArray = state.toIntArray()
            assertEquals(4, stateArray.size) // Ensure 4D state
            assert(stateArray[0] in 0 until 2)
            assert(stateArray[1] in 0 until 3)
            assert(stateArray[2] in 0 until 4)
            assert(stateArray[3] in 0 until 5)
        }
    }

    @Test
    fun `test allStates generates unique states`() {
        // Initialize a VTableD4 instance
        val vTable = VTableD4(2, 3, 4, 5)

        // Get all states
        val allStates = vTable.allStates()

        // Verify that all states are unique
        val uniqueStates = allStates.toSet()
        assertEquals(allStates.size, uniqueStates.size, "allStates should return a list of unique states.")
    }
}