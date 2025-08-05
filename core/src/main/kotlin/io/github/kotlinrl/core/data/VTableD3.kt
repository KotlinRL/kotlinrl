package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

class VTableD3(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D2>> {

    init {
        require(shape.size == 3) { "VTableD3 shape requires exactly 3 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D2>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D2>, value: Double): EnumerableValueFunction<NDArray<Int, D2>> =
        copy().also { it.base.table[state.toIntArray()] = value }

    override fun allStates(): List<NDArray<Int, D2>> =
        base.allStates().map { it.asD2Array() }

    override fun max(): Double =
        base.max()

    fun copy(): VTableD3 =
        VTableD3(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()

    fun asVTable4(vararg shape: Int): VTableD4 =
        VTableD4(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun asVTable5(vararg shape: Int): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}