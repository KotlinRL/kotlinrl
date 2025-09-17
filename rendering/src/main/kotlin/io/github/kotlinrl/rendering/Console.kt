package io.github.kotlinrl.rendering

import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import kotlin.math.*


fun printPolicyGrid(
    policy: D1Array<Int>,
    rows: Int,
    columns: Int,
    actionSymbols: Map<Int, String>
) {
    for (row in 0 until rows) {
        for (col in 0 until columns) {
            val a = policy[row * columns + col]
            val sym = actionSymbols[a]!!
            print("  ${sym} ")
        }
        println()
    }
}

fun printQTable(qTable: D2Array<Double>, rows: Int, columns: Int, format: String = "%6.2f", actionSymbols: Map<Int, String>) {
    println("Action Value Function:")

    for (row in 0 until rows) {
        for (col in 0 until columns) {
            val state = row * columns + col
            val value = qTable[state].max() ?: 0.0
            print(format.format(value) + " ")
        }
        println()
    }

    println("Policy Table:")

    for (row in 0 until rows) {
        for (col in 0 until columns) {
            val state = row * columns + col
            val action = qTable[state].argMax()
            print("  ${actionSymbols[action] ?: '?'} ")
        }
        println()
    }
}
