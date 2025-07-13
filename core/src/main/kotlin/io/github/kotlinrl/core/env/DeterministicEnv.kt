package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.space.Discrete
import io.github.kotlinrl.core.space.MultiDiscrete

interface DeterministicEnv : Env<IntArray, Int, MultiDiscrete, Discrete> {
    fun computeReward(state: IntArray, action: Int): Double

    fun simulateStep(action: Int): Transition<IntArray>
}