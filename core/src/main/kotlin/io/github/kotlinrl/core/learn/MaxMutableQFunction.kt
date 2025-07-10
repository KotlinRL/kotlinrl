package io.github.kotlinrl.core.learn

interface MaxMutableQFunction<State, Action> : MutableQFunction<State, Action> {
    fun maxValue(state: State?): Double = if (state == null) 0.0 else computeMaxValue(state)
    fun computeMaxValue(state: State): Double
}