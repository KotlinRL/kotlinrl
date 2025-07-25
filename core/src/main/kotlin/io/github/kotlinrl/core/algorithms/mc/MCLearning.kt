package io.github.kotlinrl.core.algorithms.mc

import io.github.kotlinrl.core.*

abstract class MCLearning<State, Action>(
    protected val qTable: QFunction<State, Action>,
    protected val gamma: Double,
    protected val stateActionKeyFunction: StateActionKeyFunction<State, Action>
): TrajectoryLearner<State, Action>