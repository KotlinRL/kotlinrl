package io.github.kotlinrl.core.policy

interface MutablePolicy<State, Action> : Policy<State, Action> {
    operator fun get(state: State): Action
    operator fun set(state: State, action: Action)
    fun copy(): MutablePolicy<State, Action>
}
