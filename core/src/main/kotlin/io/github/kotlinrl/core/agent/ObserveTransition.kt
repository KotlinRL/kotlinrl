package io.github.kotlinrl.core.agent

fun interface ObserveTransition<State, Action> : LearningBehavior<State, Action> {
    operator fun invoke(transition: Transition<State, Action>)
}