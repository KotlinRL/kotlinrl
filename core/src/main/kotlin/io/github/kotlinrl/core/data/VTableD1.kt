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

    override fun update(state: Int, value: Double): EnumerableValueFunction<Int> {
        val updatedBase = base.update(
            state = mk.ndarray(intArrayOf(state)).asDNArray(),
            value
        ) as VTableDN
        val new = VTableD1(*shape)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<Int> =
        base.allStates().map { it[0] }

    override fun max(): Double =
        base.max()

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()

    fun asVTable2(vararg shape: Int): VTableD2 {
        val vTable = VTableD2(*shape)
        val reshapedBase = base.table.reshape(shape[0], shape[1])
        return vTable.also {
            reshapedBase.data.copyInto(it.base.table.data)
        }
    }
}