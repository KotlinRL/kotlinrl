package io.github.kotlinrl.rendering

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.*


fun printTabularPolicy(
    policy: D1Array<Int>,
    actionSymbols: Map<Int, String>
) {
    val size = 5
    val cellW = max(1, actionSymbols.values.maxOfOrNull { it.length } ?: 1)

    for (row in 0 until size) {
        val line = StringBuilder()
        for (col in 0 until size) {
            val a = policy[row * size + col]
            val sym = actionSymbols[a]!!
            line.append(sym.padEnd(cellW)).append(' ')
        }
        println(line.toString().trimEnd())
    }
}
