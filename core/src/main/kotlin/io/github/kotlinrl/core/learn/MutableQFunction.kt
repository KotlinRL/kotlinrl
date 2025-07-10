package io.github.kotlinrl.core.learn

import io.github.kotlinrl.core.policy.*

interface MutableQFunction<State, Action> : QFunction<State, Action> {
    fun update(state: State, action: Action, newValue: Double)
}