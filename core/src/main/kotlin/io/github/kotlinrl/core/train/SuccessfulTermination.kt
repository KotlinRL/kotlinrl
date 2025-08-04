package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.agent.*

fun interface SuccessfulTermination<State, Action> {
    operator fun invoke(result: Transition<State, Action>): Boolean
}