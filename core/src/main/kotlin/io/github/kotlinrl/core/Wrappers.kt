package io.github.kotlinrl.core

import io.github.kotlinrl.core.space.Space

typealias ClipAction<State, Num, D, ObservationSpace> = io.github.kotlinrl.core.wrapper.ClipAction<State, Num, D, ObservationSpace>
typealias FilterAction<State, ObservationSpace> = io.github.kotlinrl.core.wrapper.FilterAction<State, ObservationSpace>
typealias FilterObservation<Action, ActionSpace> = io.github.kotlinrl.core.wrapper.FilterObservation<Action, ActionSpace>
typealias FlattenObservation<Num, WrappedState, Action, WrappedObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.FlattenObservation<Num, WrappedState, Action, WrappedObservationSpace, ActionSpace>
typealias Monitor<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.Monitor<State, Action, ObservationSpace, ActionSpace>
typealias NormalizeReward<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.NormalizeReward<State, Action, ObservationSpace, ActionSpace>
typealias NormalizeState<Num, D, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.NormalizeState<Num, D, Action, ObservationSpace, ActionSpace>
typealias OrderEnforcing<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.OrderEnforcing<State, Action, ObservationSpace, ActionSpace>
typealias RecordEpisodeStatistics<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RecordEpisodeStatistics<State, Action, ObservationSpace, ActionSpace>
typealias RecordVideo<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RecordVideo<State, Action, ObservationSpace, ActionSpace>
typealias RescaleAction<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.RescaleAction<State, Action, ObservationSpace, ActionSpace>
typealias RunningStats = io.github.kotlinrl.core.wrapper.RunningStats
typealias TimeLimit<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.TimeLimit<State, Action, ObservationSpace, ActionSpace>
typealias TransformReward<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.wrapper.TransformReward<State, Action, ObservationSpace, ActionSpace>
typealias TransformState<State, Action, ObservationSpace, ActionSpace, WrappedState, WrappedObservationSpace> = io.github.kotlinrl.core.wrapper.TransformState<State, Action, ObservationSpace, ActionSpace, WrappedState, WrappedObservationSpace>
