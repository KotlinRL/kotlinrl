package org.kotlinrl.core.env

import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.kotlinrl.core.space.*

interface BoxNDArrayDoubleD1BoxNDArrayFloatD1Env : Env<NDArray<Double, D1>, NDArray<Float, D1>, BoxNDArrayD1<Double>, BoxNDArrayD1<Float>>