package io.github.kotlinrl.core.algorithms.dp

import io.github.kotlinrl.core.MutablePolicy
import io.github.kotlinrl.core.Planner
import io.github.kotlinrl.core.ValueFunction

abstract class DPIteration<State, Action>(
    protected val gamma: Double = 0.99,
    protected val theta: Double = 1e-6,
    val vTable: ValueFunction<State>,
    val pTable: MutablePolicy<State, Action>
) : Planner<State, Action>