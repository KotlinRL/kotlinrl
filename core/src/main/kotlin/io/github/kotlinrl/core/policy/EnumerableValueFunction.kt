package io.github.kotlinrl.core.policy

interface EnumerableValueFunction<State> : ValueFunction<State> {
    fun max(): Double

    fun allStates(): List<State>

    override fun update(state: State, value: Double): EnumerableValueFunction<State>
}
