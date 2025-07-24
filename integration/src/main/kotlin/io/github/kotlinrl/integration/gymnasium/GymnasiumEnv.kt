package io.github.kotlinrl.integration.gymnasium

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import io.github.kotlinrl.integration.gymnasium.GymnasiumEnvs.*
import org.jetbrains.kotlinx.multik.ndarray.data.*

typealias gymnasium = GymnasiumEnv

object GymnasiumEnv {
    fun <E : Env<*, *, *, *>> make(
        envType: GymnasiumEnvs,
        seed: Int? = null,
        render: Boolean = true,
        options: Map<String, Any?> = emptyMap(),
        host: String = "localhost:50051"
    ): E {
        @Suppress("UNCHECKED_CAST")
        return when (envType) {
            CartPole_v1,
            Acrobot_v1,
            MountainCar_v0,
            LunarLander_v3 -> RemoteEnvClient<NDArray<Float, D1>, Int, Box<Float, D1>, Discrete>(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )

            FrozenLake_v1,
            CliffWalking_v0,
            Taxi_v3 -> RemoteEnvClient<Int, Int, Discrete, Discrete>(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )

            Pendulum_v1,
            BipedalWalker_v3,
            MountainCarContinuous_v0 -> RemoteEnvClient<NDArray<Float, D1>, NDArray<Float, D1>, Box<Float, D1>, Box<Float, D1>>(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )

            Ant_v5 ,
            HalfCheetah_v5,
            Hopper_v5,
            Humanoid_v5,
            HumanoidStandup_v5,
            InvertedDoublePendulum_v5,
            InvertedPendulum_v5,
            Pusher_v5,
            Reacher_v5,
            Swimmer_v5,
            Walker2d_v5 -> RemoteEnvClient<NDArray<Double, D1>, NDArray<Float, D1>, Box<Double, D1>, Box<Float, D1>>(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )

            CarRacing_v3 -> RemoteEnvClient<NDArray<Float, D1>, NDArray<Byte, D3>, Box<Float, D1>, Box<Byte, D3>>(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )

            Blackjack_v1 -> RemoteEnvClient<List<Any>, Int, Tuple, Discrete>(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )

            else -> TODO("Add support for $envType")
        } as E
    }
}