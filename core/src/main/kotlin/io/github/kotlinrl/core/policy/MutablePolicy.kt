package io.github.kotlinrl.core.policy

interface MutablePolicy<State, Action> : Policy<State, Action> {
    operator fun get(state: State): Action = invoke(state)
    operator fun set(state: State, action: Action)
}
