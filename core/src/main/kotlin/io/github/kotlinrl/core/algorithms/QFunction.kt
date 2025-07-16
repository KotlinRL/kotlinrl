package io.github.kotlinrl.core.algorithms

interface QFunction<State, Action> {
    operator fun get(state: State, action: Action): Double
    operator fun set(state: State, action: Action, value: Double)

    fun maxValue(state: State): Double

    fun bestAction(state: State): Action

    fun save(path: String)

    fun load(path: String)
}