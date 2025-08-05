package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

interface LearnableMDPModel<State, Action> : MDPModel<State, Action> {
    fun update(transition: Transition<State, Action>)
    fun sampleTransition(): Transition<State, Action>?
    fun isKnown(state: State, action: Action): Boolean
}
