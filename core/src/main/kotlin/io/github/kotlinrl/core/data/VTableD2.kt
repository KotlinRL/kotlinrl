package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTableD2(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D1>> {

    init {
        require(shape.size == 2) { "VTableD2 shape requires exactly 2 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D1>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D1>, value: Double): EnumerableValueFunction<NDArray<Int, D1>> {
        val updatedBase = base.update(state.asDNArray(), value) as VTableDN
        val new = VTableD2(*shape)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<NDArray<Int, D1>> =
        base.allStates().map { it.asD1Array() }

    override fun max(): Double =
        base.max()

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}