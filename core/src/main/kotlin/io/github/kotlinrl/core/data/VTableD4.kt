package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTableD4(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D3>> {

    init {
        require(shape.size == 4) { "VTableD4 shape requires exactly 4 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D3>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D3>, value: Double): EnumerableValueFunction<NDArray<Int, D3>> {
        val updatedBase = base.update(state.asDNArray(), value) as VTableDN
        val new = VTableD4(*shape)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<NDArray<Int, D3>> =
        base.allStates().map { it.asD3Array() }

    override fun max(): Double =
        base.max()

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}