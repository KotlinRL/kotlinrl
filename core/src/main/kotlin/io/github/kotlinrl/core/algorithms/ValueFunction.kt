package io.github.kotlinrl.core.algorithms

interface ValueFunction<State> {
    operator fun get(state: State): Double
    operator fun set(state: State, value: Double)

    fun max(): Double
    fun allStates(): List<State>
}
