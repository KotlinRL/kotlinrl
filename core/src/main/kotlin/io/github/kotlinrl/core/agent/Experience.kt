package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.env.*

data class Experience<State, Action>(
    val transition: Transition<State>,
    val action: Action,
    val state: State
)