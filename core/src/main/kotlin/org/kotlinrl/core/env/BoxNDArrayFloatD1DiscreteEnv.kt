package org.kotlinrl.core.env

import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.kotlinrl.core.space.*

interface BoxNDArrayFloatD1DiscreteEnv :Env<NDArray<Float, D1>, Int, BoxNDArrayD1<Float>, Discrete>