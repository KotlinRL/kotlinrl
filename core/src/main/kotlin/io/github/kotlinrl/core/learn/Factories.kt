package io.github.kotlinrl.core.learn

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import org.jetbrains.kotlinx.multik.ndarray.data.get

fun <State, Action> mutableQFunction(
    shape: IntArray,
    indexFor: (State, Action) -> IntArray
): MutableQFunction<State, Action> = object : MutableQFunction<State, Action> {
    val table = mk.zeros<Double, DN>(shape, DoubleDataType)

    override fun update(state: State, action: Action, newValue: Double) {
        table[indexFor(state, action)] = newValue
    }

    override fun invoke(state: State, action: Action): Double {
        return table[indexFor(state, action)]
    }
}
