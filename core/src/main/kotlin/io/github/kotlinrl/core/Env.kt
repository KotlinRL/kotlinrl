package io.github.kotlinrl.core

typealias InitialState<State> = io.github.kotlinrl.core.env.InitialState<State>
typealias Transition<State> = io.github.kotlinrl.core.env.Transition<State>
typealias Rendering = io.github.kotlinrl.core.env.Rendering
typealias RenderFrame = io.github.kotlinrl.core.env.Rendering.RenderFrame
typealias ModelBasedEnv<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.env.ModelBasedEnv<State, Action, ObservationSpace, ActionSpace>
