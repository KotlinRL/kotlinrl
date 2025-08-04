package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.*

abstract class DPIteration<State, Action> : Planner<State, Action> {

    override operator fun invoke(): Policy<State, Action> = plan()

    abstract fun plan(): Policy<State, Action>
}
