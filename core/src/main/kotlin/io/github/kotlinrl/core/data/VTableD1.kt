package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTableD1(
    vararg val shape: Int
) : EnumerableValueFunction<Int> {

    init {
        require(shape.size == 1) { "VTableD1 shape requires exactly 1 argument" }
    }

    private val base = VTableDN(shape = shape)

    override fun get(state: Int): Double =
        base[mk.ndarray(intArrayOf(state)).asDNArray()]

    override fun update(state: Int, value: Double): EnumerableValueFunction<Int> =
        copy().also { it.base.table[intArrayOf(state)] = value }

    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }

    override fun max(): Double =
        base.max()

    fun copy(): VTableD1 =
        VTableD1(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()

    fun asVTable2(vararg shape: Int): VTableD2 =
        VTableD2(*shape).also {
            base.table.data.copyInto(it.base.table.data)
        }

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