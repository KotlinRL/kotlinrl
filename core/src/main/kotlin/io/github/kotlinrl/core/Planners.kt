package io.github.kotlinrl.core

typealias ValueFunction<State> = io.github.kotlinrl.core.algorithms.ValueFunction<State>
typealias Planner<State, Action> = io.github.kotlinrl.core.plan.Planner<State, Action>
typealias TransitionFunction<State, Action> = io.github.kotlinrl.core.plan.TransitionFunction<State, Action>

fun <State, Action> policyIterationPlanner(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>
) = PolicyIteration(gamma, theta, vTable, pTable)

fun <State, Action> valueIterationPlanner(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    actionComparator: Comparator<Action>
) = ValueIteration(gamma, theta, vTable, pTable, actionComparator)