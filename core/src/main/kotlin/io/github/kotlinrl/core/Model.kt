package io.github.kotlinrl.core

typealias EmpiricalMDPModel<State, Action> = io.github.kotlinrl.core.model.EmpiricalMDPModel<State, Action>
typealias LearnableMDPModel<State, Action> = io.github.kotlinrl.core.model.LearnableMDPModel<State, Action>
typealias MDPModel<State, Action> = io.github.kotlinrl.core.model.MDPModel<State, Action>
typealias ProbabilisticTransition<State, Action> = io.github.kotlinrl.core.model.ProbabilisticTransition<State, Action>
typealias ProbabilisticTrajectory<State, Action> = List<ProbabilisticTransition<State, Action>>
