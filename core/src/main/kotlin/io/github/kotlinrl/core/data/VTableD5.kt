package io.github.kotlinrl.core.data

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

class VTableD5(
    vararg val shape: Int
) : EnumerableValueFunction<NDArray<Int, D4>> {

    init {
        require(shape.size == 4) { "VTableD5 shape requires exactly 4 arguments" }
    }

    internal val base = VTableDN(shape = shape)

    override fun get(state: NDArray<Int, D4>): Double =
        base[state.asDNArray()]

    override fun update(state: NDArray<Int, D4>, value: Double): EnumerableValueFunction<NDArray<Int, D4>> {
        val updatedBase = base.update(state.asDNArray(), value) as VTableDN
        val new = VTableD5(*shape)
        updatedBase.table.data.copyInto(new.base.table.data)
        return new
    }

    override fun allStates(): List<NDArray<Int, D4>> =
        base.allStates().map { it.asD4Array() }

    override fun max(): Double =
        base.max()

    fun save(path: String) = base.save(path)

    fun load(path: String) = base.load(path)

    fun print() = base.print()
}