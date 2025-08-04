package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTableD3(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D2>> {

    init {
        require(shape.size == 3) { "VTableD3 shape requires exactly 3 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D2>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D2>, value: Double): EnumerableValueFunction<NDArray<Int, D2>> {
        val updatedBase = base.update(state.asDNArray(), value) as VTableDN
        val new = VTableD3(*shape)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<NDArray<Int, D2>> =
        base.allStates().map { it.asD2Array() }

    override fun max(): Double =
        base.max()

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}