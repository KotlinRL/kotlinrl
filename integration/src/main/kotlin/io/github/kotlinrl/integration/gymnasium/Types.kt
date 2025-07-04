package io.github.kotlinrl.integration.gymnasium

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

typealias CartPoleEnv = Env<NDArray<Float, D1>, Int, Box<Float, D1>, Discrete>
typealias AcrobotEnv = Env<NDArray<Float, D1>, Int, Box<Float, D1>, Discrete>
typealias MountainCarEnv = Env<NDArray<Float, D1>, Int, Box<Float, D1>, Discrete>
typealias LunarLanderEnv = Env<NDArray<Float, D1>, Int, Box<Float, D1>, Discrete>

typealias FrozenLakeEnv = Env<Int, Int, Discrete, Discrete>
typealias CliffWalkingEnv = Env<Int, Int, Discrete, Discrete>
typealias TaxiEnv = Env<Int, Int, Discrete, Discrete>

typealias PendulumEnv = Env<NDArray<Float, D1>, NDArray<Float, D1>, Box<Float, D1>, Box<Float, D1>>
typealias BipedalWalkerEnv = Env<NDArray<Float, D1>, NDArray<Float, D1>, Box<Float, D1>, Box<Float, D1>>
typealias MountainCarContinuousEnv = Env<NDArray<Float, D1>, NDArray<Float, D1>, Box<Float, D1>, Box<Float, D1>>

typealias AntEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias HalfCheetahEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias HopperEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias HumanoidEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias HumanoidStandupEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias InvertedDoublePendulumEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias InvertedPendulumEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias PusherEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias ReacherEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias SwimmerEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>
typealias Walker2dEnv = Env<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>

typealias CarRacingEnv = Env<NDArray<Float, D1>, NDArray<Byte, D3>, Box<Float, D1>, Box<Byte, D3>>

typealias BlackjackEnv = Env<List<Any>, Int, Tuple, Discrete>