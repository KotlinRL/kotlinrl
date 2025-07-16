package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.*

interface ModelBasedEnv<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>
    : Env<State, Action, ObservationSpace, ActionSpace> {

    fun simulateStep(state: State, action: Action): Transition<State>
}