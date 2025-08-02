package io.github.kotlinrl.core.policy

import io.github.kotlinrl.core.Policy

fun interface Planner<State, Action> {
    operator fun invoke(): Policy<State, Action>
}