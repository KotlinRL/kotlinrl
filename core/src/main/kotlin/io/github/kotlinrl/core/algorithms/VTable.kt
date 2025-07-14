package io.github.kotlinrl.core.algorithms

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTable(private val size: Int) {
    private val values: D2Array<Double> = mk.d2array(size, size) { 0.0 }

    fun max(): Double = values.data.max()

    fun allStates(): Sequence<IntArray> = sequence {
        for (i in 0 until size) {
            for (j in 0 until size) {
                yield(intArrayOf(i, j))
            }
        }
    }
    operator fun get(state: IntArray): Double = values[state[0], state[1]]
    operator fun set(state: IntArray, value: Double) {
        values[state[0], state[1]] = value
    }
}