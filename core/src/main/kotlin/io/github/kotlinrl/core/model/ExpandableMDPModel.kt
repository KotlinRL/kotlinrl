package io.github.kotlinrl.core.model

import io.github.kotlinrl.core.*

interface ExpandableMDPModel<State, Action> {
    fun predecessors(state: State): Set<StateActionKey<*, *>>
    fun visitCount(state: State, action: Action): Int
    fun isTerminal(state: State): Boolean
}
