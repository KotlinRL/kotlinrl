package io.github.kotlinrl.core.agent

interface StepCallback<State, Action> {
    fun beforeStep(state: State) { }
    fun afterStep(state: State, action: Action) { }
}