package org.kotlinrl.integration.gymnasium

import org.kotlinrl.core.env.*
import org.kotlinrl.core.space.*
import org.kotlinrl.integration.gymnasium.GymnasiumEnvs.*

typealias gymnasium = GymnasiumEnv
object GymnasiumEnv {
    fun <
            Observation,
            Action,
            Reward,
            OS: Space<Observation>,
            AS : Space<Action>,
            E : Env<Observation, Action, Reward, OS, AS>>
            make(envType: GymnasiumEnvs,
                 seed: Int? = null,
                 render: Boolean = true,
                 options: Map<String, String> = emptyMap(),
                 host: String = "localhost:50051"
    ): E {
        @Suppress("UNCHECKED_CAST")
        return when(envType) {
            CartPole_v1,
            Acrobot_v1,
            MountainCar_v0,
            LunarLander_v3,
            MountatCarContinuous_v0 -> RemoteFloatBoxNDArrayD1DiscreteEnv(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )
            FrozenLake_v1,
            CliffWalking_v0,
            Taxi_v3 -> RemoteFloatDiscreteDiscreteEnv(
                envName = envType.envName,
                seed = seed,
                render = render,
                options = options,
                host = host
            )
            Pendulum_v1,
            Ant_v5,
            BipedalWalker_v3 -> RemoteFloatBoxNDArrayD1BoxNDArrayD1Env(
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