package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray

class VTableD2(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D1>> {

    init {
        require(shape.size == 2) { "VTableD2 shape requires exactly 2 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D1>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D1>, value: Double): EnumerableValueFunction<NDArray<Int, D1>> =
        copy().also { it.base.table[state.toIntArray()] = value }


    override fun allStates(): List<NDArray<Int, D1>> =
        base.allStates().map { it.asD1Array() }

    override fun max(): Double =
        base.max()

    fun copy(): VTableD2 =
        VTableD2(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()

    fun asVTable3(vararg shape: Int): VTableD3 =
        VTableD3(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

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