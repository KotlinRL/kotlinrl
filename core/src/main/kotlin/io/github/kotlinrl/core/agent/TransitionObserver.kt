package io.github.kotlinrl.core.agent

fun interface TransitionObserver<State, Action> {
    operator fun invoke(transition: Transition<State, Action>)
}