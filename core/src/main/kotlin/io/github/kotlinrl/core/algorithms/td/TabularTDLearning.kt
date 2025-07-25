package io.github.kotlinrl.core.algorithms.td

import io.github.kotlinrl.core.*

abstract class TabularTDLearning<State, Action>(
    protected val qTable: QFunction<State, Action>,
    protected val alpha: ParameterSchedule,
    protected val gamma: Double
) : TransitionLearner<State, Action>
