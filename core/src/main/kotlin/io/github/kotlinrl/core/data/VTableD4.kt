package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

class VTableD4(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D3>> {

    init {
        require(shape.size == 4) { "VTableD4 shape requires exactly 4 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D3>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D3>, value: Double): EnumerableValueFunction<NDArray<Int, D3>> =
        copy().also { it.base.table[state.toIntArray()] = value }

    override fun allStates(): List<NDArray<Int, D3>> =
        base.allStates().map { it.asD3Array() }

    override fun max(): Double =
        base.max()

    fun copy(): VTableD4 =
        VTableD4(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()

    fun asVTable5(vararg shape: Int): VTableD5 =
        VTableD5(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun asVTableN(vararg shape: Int): VTableDN =
        VTableDN(*shape).also {
            base.table.data.copyInto(it.table.data)
        }
}