package io.github.kotlinrl.core.agent

interface StateActionCallback<State, Action> {
    fun before(state: State) { }
    fun after(state: State, action: Action) { }
}