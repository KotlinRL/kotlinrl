package io.github.kotlinrl.core

typealias InitialState<State> = io.github.kotlinrl.core.env.InitialState<State>
typealias StepResult<State> = io.github.kotlinrl.core.env.StepResult<State>
typealias Rendering = io.github.kotlinrl.core.env.Rendering
typealias RenderFrame = io.github.kotlinrl.core.env.Rendering.RenderFrame
typealias ModelBasedEnv<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.env.ModelBasedEnv<State, Action, ObservationSpace, ActionSpace>
typealias Env<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.env.Env<State, Action, ObservationSpace, ActionSpace>
