package org.kotlinrl.core.env

import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.kotlinrl.core.space.*

interface FloatBoxNDArrayD1DiscreteEnv :Env<NDArray<Float, D1>, Int, Float, BoxNDArrayD1<Float>, Discrete>