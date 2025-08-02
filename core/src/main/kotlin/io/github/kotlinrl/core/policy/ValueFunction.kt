package io.github.kotlinrl.core.policy

interface ValueFunction<State> {
    operator fun get(state: State): Double

    fun update(state: State, value: Double): ValueFunction<State>
}
