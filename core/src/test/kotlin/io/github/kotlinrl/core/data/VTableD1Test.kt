package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class VTableD1Test {

    /**
     * The `VTableD1` class represents a one-dimensional value function table.
     * The `allStates` method retrieves all possible states as a list of integers.
     */

    @Test
    fun `test allStates with empty table`() {
        // Initialize a VTableD1 with no states
        val vTable = VTableD1(1)

        // Run the allStates function and assert the result is an empty list
        val result = vTable.allStates()
        assertEquals(
            listOf(0),
            result,
            "Expected allStates to return an empty list for a table with zero states."
        )
    }

    @Test
    fun `test allStates with single state`() {
        // Initialize a VTableD1 with 1 state
        val vTable = VTableD1(1)

        // Run the allStates function and assert the result contains a single state [0]
        val result = vTable.allStates()
        assertEquals(listOf(0), result, "Expected allStates to return [0] for a table with one state.")
    }

    @Test
    fun `test Load Functionality`() {
        // Initialize and save a VTableD1 with 3 states
        val initialVTable = VTableD1(3)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)

        val path = "test_load_vtable.txt"
        initialVTable.save(path)

        // Create a new VTableD1 instance and load the saved data
        val loadedVTable = VTableD1(3)
        loadedVTable.load(path)

        // Verify that the loaded data matches the saved data
        assertEquals(initialVTable.allStates(), loadedVTable.allStates())
        for (state in initialVTable.allStates()) {
            assertEquals(
                initialVTable.get(state),
                loadedVTable.get(state),
                "State $state does not match in the loaded VTable."
            )
        }
    }

    @Test
    fun `test Load Into Existing VTable`() {
        // Initialize a VTableD1 with 3 states and custom values
        val initialVTable = VTableD1(3)
            .update(0, 4.0)
            .update(1, 5.0)
            .update(2, 6.0)

        // Save the initial VTable state
        val path = "test_load_existing_vtable.txt"
        initialVTable.save(path)

        // Create an existing VTableD1 with different values and load the saved data into it
        val existingVTable = VTableD1(3)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)

        existingVTable.load(path)

        // Verify that the existing VTable matches the saved VTable data
        for (state in initialVTable.allStates()) {
            assertEquals(
                initialVTable.get(state),
                existingVTable.get(state),
                "State $state does not match after loading into an existing VTable."
            )
        }
    }

    @Test
    fun `test allStates with multiple states`() {
        // Initialize a VTableD1 with 3 states
        val vTable = VTableD1(3)

        // Run the allStates function and assert the result contains all states [0, 1, 2]
        val result = vTable.allStates()
        assertEquals(listOf(0, 1, 2), result, "Expected allStates to return [0, 1, 2] for a table with three states.")
    }

    @Test
    fun `test allStates with high number of states`() {
        // Initialize a VTableD1 with 1000 states
        val numberOfStates = 1000
        val vTable = VTableD1(numberOfStates)

        // Run the allStates function and assert the result contains all states from 0 to 999
        val result = vTable.allStates()
        assertEquals(
            (0 until numberOfStates).toList(),
            result,
            "Expected allStates to return all states from 0 to 999."
        )
    }

    @Test
    fun `test allStates after updating table`() {
        // Initialize a VTableD1 with 5 states
        val vTable = VTableD1(5)

        // Update one state's value
        vTable.update(3, 10.0)

        // Run the allStates function and assert the result still contains all states [0, 1, 2, 3, 4]
        val result = vTable.allStates()
        assertEquals(
            listOf(0, 1, 2, 3, 4),
            result,
            "Expected allStates to return [0, 1, 2, 3, 4] after updating a state value."
        )
    }

    @Test
    fun `test Max With Empty Table`() {
        // Initialize a VTableD1 with no states
        val vTable = VTableD1(1)

        assertEquals(0.0, vTable.max())
    }

    @Test
    fun `test Max With Single State`() {
        // Initialize a VTableD1 with 1 state
        val vTable = VTableD1(1)
            .update(0, 5.0) // Set the value of the single state

        // Run the max function and assert the result
        val result = vTable.max()
        assertEquals(5.0, result, "Expected max to return the value of the single state.")
    }

    @Test
    fun `test Max With Multiple States`() {
        // Initialize a VTableD1 with 3 states
        val vTable = VTableD1(3)
            .update(0, 1.0)
            .update(1, 3.0)
            .update(2, 2.0)

        // Run the max function and assert the result
        val result = vTable.max()
        assertEquals(3.0, result, "Expected max to return the maximum value in the table.")
    }

    @Test
    fun `test Max After Updating Table`() {
        // Initialize a VTableD1 with 4 states
        var vTable = VTableD1(4)
            .update(0, 2.0)
            .update(1, 4.0)
            .update(2, 3.0)
            .update(3, 1.0)

        // Update one state's value to a higher value
        vTable = vTable.update(3, 5.0)

        // Run the max function and assert the result
        val result = vTable.max()
        assertEquals(5.0, result, "Expected max to return the updated maximum value in the table.")
    }

    @Test
    fun `test Save Functionality`() {
        // Initialize a VTableD1 with 3 states
        val vTable = VTableD1(3)
            .update(0, 1.0)
            .update(1, 3.0)
            .update(2, 2.0)

        // Path to save the VTableD1 data
        val path = "test_save_vtable.txt"

        // Save the VTableD1 data to the specified path
        vTable.save(path)

        // Load the saved data into a new instance
        val loadedVTable = VTableD1(3)
        loadedVTable.load(path)

        // Assert that the states and their values are the same in the loaded instance
        assertEquals(vTable.allStates(), loadedVTable.allStates())
        for (state in vTable.allStates()) {
            assertEquals(
                vTable.get(state),
                loadedVTable.get(state),
                "State $state does not match in the loaded VTable."
            )
        }
    }

    @Test
    fun `test Print With Empty Table`() {
        // Initialize a VTableD1 with no states
        val vTable = VTableD1(1).update(0, 1.0)

        // Redirect the standard output
        val outputStream = java.io.ByteArrayOutputStream()
        val printStream = java.io.PrintStream(outputStream)
        val originalOut = System.out
        System.setOut(printStream)

        // Execute the print function
        vTable.print()

        // Restore the standard output
        System.setOut(originalOut)

        // Verify the output
        val actualOutput = outputStream.toString().trim()
        assertEquals("[1.0]", actualOutput, "Expected printed output to be '[1.0]' for an empty table.")
    }

    @Test
    fun `test Print With Populated Table`() {
        // Initialize a VTableD1 with 3 states
        val vTable = VTableD1(3)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)

        // Redirect the standard output
        val outputStream = java.io.ByteArrayOutputStream()
        val printStream = java.io.PrintStream(outputStream)
        val originalOut = System.out
        System.setOut(printStream)

        // Execute the print function
        vTable.print()

        // Restore the standard output
        System.setOut(originalOut)

        // Verify the output
        val actualOutput = outputStream.toString().trim()
        assertEquals("[1.0, 2.0, 3.0]", actualOutput, "Expected printed output to match the table's state values.")
    }

    @Test
    fun `test asVTable4 with valid dimensions`() {
        // Initialize a VTableD1 with 24 states
        val vTableD1 = VTableD1(24)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)
            .update(3, 4.0)
            .update(4, 5.0)
            .update(5, 6.0)
            .update(6, 7.0)
            .update(7, 8.0)
            .update(8, 9.0)
            .update(9, 10.0)
            .update(10, 11.0)
            .update(11, 12.0)
            .update(12, 13.0)
            .update(13, 14.0)
            .update(14, 15.0)
            .update(15, 16.0)
            .update(16, 17.0)
            .update(17, 18.0)
            .update(18, 19.0)
            .update(19, 20.0)
            .update(20, 21.0)
            .update(21, 22.0)
            .update(22, 23.0)
            .update(23, 24.0)

        // Convert it to a VTableD4 with dimensions 2x3x2x2
        val vTableD4 = vTableD1.asVTable4(2, 3, 2, 2)

        // Verify all states and values are properly transferred
        assertEquals(2, vTableD4.shape[0])
        assertEquals(3, vTableD4.shape[1])
        assertEquals(2, vTableD4.shape[2])
        assertEquals(2, vTableD4.shape[3])
        assertEquals(1.0, vTableD4[0, 0, 0, 0])
        assertEquals(2.0, vTableD4[0, 0, 0, 1])
        assertEquals(3.0, vTableD4[0, 0, 1, 0])
        assertEquals(4.0, vTableD4[0, 0, 1, 1])
        assertEquals(5.0, vTableD4[0, 1, 0, 0])
        assertEquals(6.0, vTableD4[0, 1, 0, 1])
        assertEquals(7.0, vTableD4[0, 1, 1, 0])
        assertEquals(24.0, vTableD4[1, 2, 1, 1])
    }

    @Test
    fun `test asVTable3 with valid dimensions`() {
        // Initialize a VTableD1 with 12 states
        val vTableD1 = VTableD1(12)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)
            .update(3, 4.0)
            .update(4, 5.0)
            .update(5, 6.0)
            .update(6, 7.0)
            .update(7, 8.0)
            .update(8, 9.0)
            .update(9, 10.0)
            .update(10, 11.0)
            .update(11, 12.0)

        // Convert it to a VTableD3 with 2 rows, 3 columns, and 2 layers
        val vTableD3 = vTableD1.asVTable3(2, 3, 2)

        // Verify all states and values are properly transferred
        assertEquals(2, vTableD3.shape[0])
        assertEquals(3, vTableD3.shape[1])
        assertEquals(2, vTableD3.shape[2])
        assertEquals(1.0, vTableD3[0, 0, 0])
        assertEquals(2.0, vTableD3[0, 0, 1])
        assertEquals(3.0, vTableD3[0, 1, 0])
        assertEquals(4.0, vTableD3[0, 1, 1])
        assertEquals(5.0, vTableD3[0, 2, 0])
        assertEquals(6.0, vTableD3[0, 2, 1])
        assertEquals(7.0, vTableD3[1, 0, 0])
        assertEquals(8.0, vTableD3[1, 0, 1])
        assertEquals(9.0, vTableD3[1, 1, 0])
        assertEquals(10.0, vTableD3[1, 1, 1])
        assertEquals(11.0, vTableD3[1, 2, 0])
        assertEquals(12.0, vTableD3[1, 2, 1])
    }

    @Test
    fun `test asVTable2 with valid dimensions`() {
        // Initialize a VTableD1 with 6 states
        val vTableD1 = VTableD1(6)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)
            .update(3, 4.0)
            .update(4, 5.0)
            .update(5, 6.0)

        // Convert it to a VTableD2 with 2 rows and 3 columns
        val vTableD2 = vTableD1.asVTable2(2, 3)

        // Verify all states and values are properly transferred
        assertEquals(2, vTableD2.shape[0])
        assertEquals(3, vTableD2.shape[1])
        assertEquals(1.0, vTableD2[0, 0])
        assertEquals(2.0, vTableD2[0, 1])
        assertEquals(3.0, vTableD2[0, 2])
        assertEquals(4.0, vTableD2[1, 0])
        assertEquals(5.0, vTableD2[1, 1])
        assertEquals(6.0, vTableD2[1, 2])
    }

    @Test
    fun `test asVTable5 with valid dimensions`() {
        // Initialize a VTableD1 with 120 states
        val vTableD1 = VTableD1(120)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)
            .update(3, 4.0)
            .update(4, 5.0)
            .update(5, 6.0)
            .update(6, 7.0)
            .update(7, 8.0)
            .update(8, 9.0)
            .update(9, 10.0)
            .update(10, 11.0)
            .update(11, 12.0)
            .update(12, 13.0)
            .update(13, 14.0)
            .update(14, 15.0)
            .update(15, 16.0)
            .update(16, 17.0)
            .update(17, 18.0)
            .update(18, 19.0)
            .update(19, 20.0)
            .update(20, 21.0)
            .update(21, 22.0)
            .update(22, 23.0)
            .update(23, 24.0)

        // Convert it to a VTableD5 with dimensions 2x3x2x2x5
        val vTableD5 = vTableD1.asVTable5(2, 3, 2, 2, 5)

        // Verify all states and values are properly transferred
        assertEquals(2, vTableD5.shape[0])
        assertEquals(3, vTableD5.shape[1])
        assertEquals(2, vTableD5.shape[2])
        assertEquals(2, vTableD5.shape[3])
        assertEquals(5, vTableD5.shape[4])
        assertEquals(1.0, vTableD5[0, 0, 0, 0, 0])
        assertEquals(24.0, vTableD5[0, 1, 0, 0, 3])
    }

    @Test
    fun `test asVTableN with valid dimensions`() {
        // Initialize a VTableD1 with 24 states
        val vTableD1 = VTableD1(24)
            .update(0, 1.0)
            .update(1, 2.0)
            .update(2, 3.0)
            .update(3, 4.0)
            .update(4, 5.0)
            .update(5, 6.0)
            .update(6, 7.0)
            .update(7, 8.0)
            .update(8, 9.0)
            .update(9, 10.0)
            .update(10, 11.0)
            .update(11, 12.0)
            .update(12, 13.0)
            .update(13, 14.0)
            .update(14, 15.0)
            .update(15, 16.0)
            .update(16, 17.0)
            .update(17, 18.0)
            .update(18, 19.0)
            .update(19, 20.0)
            .update(20, 21.0)
            .update(21, 22.0)
            .update(22, 23.0)
            .update(23, 24.0)

        // Convert it to a VTableDN with valid dimensions (2x3x4)
        val vTableDN = vTableD1.asVTableN(2, 3, 4)

        // Verify all states and values are properly transferred
        assertEquals(listOf(2, 3, 4), vTableDN.shape.toList())
        assertEquals(1.0, vTableDN[mk.ndarray(intArrayOf(0, 0, 0)).asDNArray()])
        assertEquals(24.0, vTableDN[mk.ndarray(intArrayOf(1, 2, 3)).asDNArray()])
    }
}