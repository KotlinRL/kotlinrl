package io.github.kotlinrl.core

typealias Planner<State, Action> = io.github.kotlinrl.core.plan.Planner<State, Action>
typealias TransitionFunction<State, Action> = io.github.kotlinrl.core.plan.TransitionFunction<State, Action>

fun policyIterationPlanner(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: VTable,
    pTable: PTable
) = PolicyIteration(gamma, theta, vTable, pTable)

fun valueIterationPlanner(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: VTable,
    pTable: PTable
) = ValueIteration(gamma, theta, vTable, pTable)