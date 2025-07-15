package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.space.Discrete
import io.github.kotlinrl.core.space.MultiDiscrete

interface ModelBasedEnv : Env<IntArray, Int, MultiDiscrete, Discrete> {
    fun nextState(state: IntArray, action: Int): IntArray

    fun computeReward(state: IntArray, action: Int): Double

    fun simulateStep(action: Int): Transition<IntArray>
}