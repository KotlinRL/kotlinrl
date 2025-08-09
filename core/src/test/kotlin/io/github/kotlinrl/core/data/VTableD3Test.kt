package io.github.kotlinrl.core.data

import org.jetbrains.kotlinx.multik.api.*
import kotlin.test.*

class VTableD3Test {

    @Test
    fun `test asVTableN creates an instance with correct shape`() {
        val vTable = VTableD3(2, 2, 2)
        val vTableN = vTable.asVTableN(2, 2, 2, 5)

        assertEquals(
            intArrayOf(2, 2, 2, 5).toList(),
            vTableN.shape.toList(),
            "The created VTableDN should match the expected dimensions."
        )
    }

    @Test
    fun `test asVTableN retains data correctly`() {
        val vTable = VTableD3(2, 2, 2)
            .update(0, 0, 0, 7.0)
            .update(1, 1, 1, 3.5)
        val vTableN = vTable.asVTableN(2, 2, 2)

        assertEquals(
            7.0,
            vTableN[mk.ndarray(mk[mk[0, 0, 0]]).asDNArray()],
            "The value at (0, 0, 0) in the new VTableDN should match the original."
        )
        assertEquals(
            3.5,
            vTableN[mk.ndarray(mk[mk[1, 1, 1]]).asDNArray()],
            "The value at (1, 1, 1) in the new VTableDN should match the original."
        )
    }

    @Test
    fun `test asVTableN updates do not affect original`() {
        val vTable = VTableD3(2, 2, 2).update(0, 0, 0, 5.0)
        val vTableN = vTable.asVTableN(2, 2, 2, 3)
            .update(mk.ndarray(mk[mk[0, 0, 0, 0]]).asDNArray(), 10.0)

        assertEquals(
            5.0,
            vTable[0, 0, 0],
            "Updates in VTableDN created by asVTableN should not affect the original VTableD3."
        )
    }

    @Test
    fun `test asVTableN default values`() {
        val vTable = VTableD3(2, 2, 2)
        val vTableN = vTable.asVTableN(2, 2, 2, 3)

        assertEquals(
            0.0,
            vTableN[mk.ndarray(mk[mk[1, 1, 1, 0]]).asDNArray()],
            "New dimensions in VTableDN should be initialized to the default value 0.0."
        )
    }

    @Test
    fun `test save function persists data correctly to a file and reloads`() {
        val vTable = VTableD3(2, 2, 2).update(0, 0, 0, 9.0).update(1, 1, 1, 4.5)
        val filePath = "temp_vtable_save_test.txt"

        vTable.save(filePath)
        val reloadedVTable = VTableD3(2, 2, 2)
        reloadedVTable.load(filePath)

        assertEquals(9.0, reloadedVTable[0, 0, 0], "Reloaded VTable should retain updated value 9.0 at (0, 0, 0).")
        assertEquals(4.5, reloadedVTable[1, 1, 1], "Reloaded VTable should retain updated value 4.5 at (1, 1, 1).")
        assertEquals(0.0, reloadedVTable[0, 1, 1], "Reloaded VTable should retain default value 0.0 at (0, 1, 1).")
    }

    @Test
    fun `test save function creates file successfully`() {
        val vTable = VTableD3(3, 3, 3)
        val filePath = "temp_vtable_creation_test.txt"

        vTable.save(filePath)

        val file = java.io.File(filePath)
        assert(file.exists()) { "The file should be created successfully by the save function." }

        // Clean up the file after the test
        file.delete()
    }

    @Test
    fun `test asVTable5 creates an instance with correct shape`() {
        val vTable = VTableD3(2, 2, 2)
        val vTable5 = vTable.asVTable5(2, 2, 2, 4, 3)

        assertEquals(
            intArrayOf(2, 2, 2, 4, 3).toList(),
            vTable5.shape.toList(),
            "The created VTableD5 should have the expected shape."
        )
    }

    @Test
    fun `test asVTable5 updates do not affect original`() {
        val vTable = VTableD3(2, 2, 2).update(0, 0, 0, 6.0)
        val vTable5 = vTable.asVTable5(2, 2, 2, 3, 2).update(0, 0, 0, 0, 0, 12.0)

        assertEquals(
            6.0,
            vTable[0, 0, 0],
            "Updates in VTableD5 created by asVTable5 should not affect the original VTableD3."
        )
    }

    @Test
    fun `test asVTable5 default values`() {
        val vTable = VTableD3(2, 2, 2)
        val vTable5 = vTable.asVTable5(2, 2, 2, 3, 2)

        assertEquals(
            0.0,
            vTable5[1, 1, 1, 0, 0],
            "New dimensions in VTableD5 should be initialized to the default value 0.0."
        )
    }

    @Test
    fun `test max on default initialized table returns zero`() {
        val vTable = VTableD3(2, 2, 2)

        assertEquals(0.0, vTable.max(), "The max value of a default initialized VTableD3 should be 0.0.")
    }

    @Test
    fun `test max after specific updates`() {
        val vTable = VTableD3(3, 3, 3)
            .update(0, 0, 0, 10.0)
            .update(1, 1, 1, 20.0)
            .update(2, 2, 2, 5.0)

        assertEquals(20.0, vTable.max(), "The max value of the VTableD3 should be the largest updated value.")
    }

    @Test
    fun `test max on table with identical values`() {
        var vTable = VTableD3(2, 2, 2)
        vTable = vTable.update(0, 0, 0, 7.0)
        vTable = vTable.update(1, 1, 1, 7.0)

        assertEquals(7.0, vTable.max(), "The max value should be correct when all table entries are identical.")
    }

    @Test
    fun `test allStates includes all possible states`() {
        val rowSize = 2
        val colSize = 2
        val layerSize = 2
        val vTable = VTableD3(rowSize, colSize, layerSize)

        val allStates = vTable.allStates()
        val expectedStateCount = rowSize * colSize * layerSize

        assertEquals(
            expectedStateCount,
            allStates.size,
            "The number of states produced by allStates() should match total state count."
        )
    }

    @Test
    fun `test allStates produces unique states`() {
        val vTable = VTableD3(2, 2, 2)

        val allStates = vTable.allStates()
        val uniqueStates = allStates.distinct()

        assertEquals(
            allStates.size,
            uniqueStates.size,
            "allStates() should produce distinct, non-duplicate states."
        )
    }

    @Test
    fun `test allStates matches expected shape`() {
        val vTable = VTableD3(3, 3, 1)

        val allStates = vTable.allStates()
        val firstStateShape = allStates.first().shape

        assertEquals(
            intArrayOf(1, 3).toList(),
            firstStateShape.toList(),
            "Each state should have the correct shape of [1, 3]."
        )
    }

    @Test
    fun `test update with NDArray updates correct value`() {
        var vTable = VTableD3(2, 2, 2)
        val updatedValue = 42.0

        vTable = vTable.update(1, 1, 0, updatedValue)

        assertEquals(updatedValue, vTable[1, 1, 0], "The value at the given state should be updated correctly.")
    }

    @Test
    fun `test update with row, col, layer updates correct value`() {
        var vTable = VTableD3(3, 3, 3)
        val row = 2
        val col = 1
        val layer = 0
        val updatedValue = 77.0

        vTable = vTable.update(row, col, layer, updatedValue)

        assertEquals(
            updatedValue,
            vTable[row, col, layer],
            "The value at the given indices should be updated correctly."
        )
    }

    @Test
    fun `test update does not affect other states`() {
        var vTable = VTableD3(2, 2, 2)
        val updatedValue = 10.0
        vTable = vTable.update(0, 0, 1, updatedValue)

        assertEquals(0.0, vTable[1, 1, 1], "States other than the updated one should remain unaffected.")
    }

    @Test
    fun `test update with row, col, layer retains correct shape`() {
        val vTable = VTableD3(4, 4, 4)
            .update(3, 3, 3, 100.0)

        assertEquals(100.0, vTable[3, 3, 3], "The value at the updated indices should be correct.")
        assertEquals(0.0, vTable[0, 0, 0], "Other values should remain unchanged.")
    }

    @Test
    fun `test update preserves immutability of original instance`() {
        val vTable = VTableD3(2, 2, 2)
        val updatedValue = 5.0

        val newVTable = vTable.update(1, 1, 1, updatedValue)

        assertEquals(0.0, vTable[1, 1, 1], "Original instance should remain unchanged.")
        assertEquals(updatedValue, newVTable[1, 1, 1], "New instance should reflect the updated value.")
    }

    @Test
    fun `test print produces expected output`() {
        val vTable = VTableD3(2, 2, 2).update(0, 0, 0, 1.5).update(1, 1, 1, 2.5)

        // Capture output from the print function
        val output = captureOutput {
            vTable.print()
        }

        assertTrue(
            output.contains("1.5"),
            "The print output should include the updated value 1.5."
        )
        assertTrue(
            output.contains("2.5"),
            "The print output should include the updated value 2.5."
        )
    }

    @Test
    fun `test print handles empty or default initialized table`() {
        val vTable = VTableD3(2, 2, 2)

        // Ensure no exception is thrown and output is captured
        val output = captureOutput {
            vTable.print()
        }

        assertTrue(
            output.isNotEmpty(),
            "The print output for a default table should not be empty."
        )
    }

    private fun captureOutput(block: () -> Unit): String {
        val outputStream = java.io.ByteArrayOutputStream()
        val printStream = java.io.PrintStream(outputStream)
        val originalOut = System.out
        System.setOut(printStream)
        block()
        System.out.flush()
        System.setOut(originalOut)
        return outputStream.toString()
    }

    @Test
    fun `test asVTable4 creates an instance with correct shape`() {
        val vTable = VTableD3(2, 2, 2)
        val vTable4 = vTable.asVTable4(2, 2, 2, 4)

        assertEquals(
            intArrayOf(2, 2, 2, 4).toList(),
            vTable4.shape.toList(),
            "The created VTableD4 should have the expected shape."
        )
    }

    @Test
    fun `test asVTable4 updates do not affect original`() {
        val vTable = VTableD3(2, 2, 2).update(0, 0, 0, 5.0)
        val vTable4 = vTable.asVTable4(2, 2, 2, 3).update(0, 0, 0, 0, 10.0)

        assertEquals(
            5.0,
            vTable[0, 0, 0],
            "Updates in VTableD4 created by asVTable4 should not affect the original VTableD3."
        )
    }

    @Test
    fun `test asVTable4 default values`() {
        val vTable = VTableD3(2, 2, 2)
        val vTable4 = vTable.asVTable4(2, 2, 2, 3)

        assertEquals(
            0.0,
            vTable4[1, 1, 1, 0],
            "New dimensions in VTableD4 should be initialized to the default value 0.0."
        )
    }
}