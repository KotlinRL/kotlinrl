package io.github.kotlinrl.core.learn

import io.github.kotlinrl.core.policy.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.io.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import java.io.*

class QTable<State, Action>(
    private val shape: IntArray,
    private val indexFor: (State, Action) -> IntArray,
    private val actionListProvider: StateActionListProvider<State, Action>
) {
    private var table = mk.zeros<Double, DN>(shape, DoubleDataType).asDNArray()

    operator fun get(state: State, action: Action): Double =
        table[indexFor(state, action)]

    operator fun set(state: State, action: Action, value: Double) {
        table[indexFor(state, action)] = value
    }

    fun maxValue(state: State): Double {
        return actionListProvider(state).maxOfOrNull { this[state, it] } ?: 0.0
    }

    fun bestAction(state: State): Action? {
        return actionListProvider(state).maxByOrNull { this[state, it] }
    }

    fun save(path: String) {
        mk.write(File(path), table.asD2Array())
    }

    fun load(path: String) {
        table = mk.read<Double, D2>(File(path)).unsqueeze(shape.size - 1).asDNArray()
    }
}
