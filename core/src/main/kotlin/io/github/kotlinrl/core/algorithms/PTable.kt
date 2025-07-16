package io.github.kotlinrl.core.algorithms

import io.github.kotlinrl.core.policy.MutablePolicy
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class PTable(
    val shape: IntArray,
    private val defaultAction: Int = 0
): MutablePolicy<IntArray, Int> {
    private val policy: NDArray<Int, DN> = mk.dnarray<Int, DN>(shape) { defaultAction }.asDNArray()

    override fun invoke(state: IntArray): Int = this[state]

    override operator fun get(state: IntArray): Int = policy[state]
    override operator fun set(state: IntArray, action: Int) {
        policy[state] = action
    }

    fun copy(): PTable {
        val copy = PTable(shape)
        policy.data.copyInto(copy.policy.data)
        return copy
    }
}