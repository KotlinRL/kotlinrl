package io.github.kotlinrl.core.algorithms

interface QFunction<State, Action, QValue> {
    operator fun get(state: State, action: Action): Double
    operator fun set(state: State, action: Action, value: Double)

    fun qValues(index: State): QValue

    fun maxValue(state: State): Double

    fun bestAction(state: State): Action

    fun save(path: String)

    fun load(path: String)
}