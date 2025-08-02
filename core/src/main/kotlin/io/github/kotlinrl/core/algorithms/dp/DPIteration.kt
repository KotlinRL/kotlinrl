package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.Planner
import io.github.kotlinrl.core.Policy
import io.github.kotlinrl.core.StateActionListProvider
import io.github.kotlinrl.core.policy.EnumerableValueFunction

abstract class DPIteration<State, Action> : Planner<State, Action> {

    override operator fun invoke(): Policy<State, Action> = plan()

    abstract fun plan(): Policy<State, Action>
}
