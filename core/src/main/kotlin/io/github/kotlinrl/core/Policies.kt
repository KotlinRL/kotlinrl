package io.github.kotlinrl.core

import kotlin.random.*

typealias ParameterSchedule = io.github.kotlinrl.core.policy.ParameterSchedule
typealias RandomPolicy<State, Action> = io.github.kotlinrl.core.policy.RandomPolicy<State, Action>
typealias GreedyPolicy<State, Action> = io.github.kotlinrl.core.policy.GreedyPolicy<State, Action>
typealias EpsilonGreedyPolicy<State, Action> = io.github.kotlinrl.core.policy.EpsilonGreedyPolicy<State, Action>
typealias SoftmaxPolicy<State, Action> = io.github.kotlinrl.core.policy.SoftmaxPolicy<State, Action>
typealias EpsilonSoftPolicy<State, Action> = io.github.kotlinrl.core.policy.EpsilonSoftPolicy<State, Action>
typealias Policy<State, Action> = io.github.kotlinrl.core.policy.Policy<State, Action>
typealias QFunction<State, Action> = io.github.kotlinrl.core.policy.QFunction<State, Action>
typealias QFunctionPolicy<State, Action> = io.github.kotlinrl.core.policy.QFunctionPolicy<State, Action>
typealias EnumerableQFunction<State, Action> = io.github.kotlinrl.core.policy.EnumerableQFunction<State, Action>
typealias ValueFunction<State> = io.github.kotlinrl.core.policy.ValueFunction<State>
typealias Planner<State, Action> = io.github.kotlinrl.core.policy.Planner<State, Action>
typealias EnumerableValueFunction<State> = io.github.kotlinrl.core.policy.EnumerableValueFunction<State>
typealias StochasticPolicy<State, Action> = io.github.kotlinrl.core.policy.StochasticPolicy<State, Action>
typealias UniformStochasticPolicy<State, Action> = io.github.kotlinrl.core.policy.UniformStochasticPolicy<State, Action>
typealias StateActions<State, Action> = io.github.kotlinrl.core.policy.StateActions<State, Action>
typealias PolicyUpdate<State, Action> = (Policy<State, Action>) -> Unit
typealias EligibilityTraceUpdate<State, Action> = (EligibilityTrace<State, Action>) -> Unit

fun <State, Action> randomPolicy(
    stateActions: StateActions<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(stateActions, rng)

fun <State, Action> greedyPolicy(
    q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    ): QFunctionPolicy<State, Action> = GreedyPolicy(q, stateActions)

fun <State, Action> epsilonGreedyPolicy(
    q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): QFunctionPolicy<State, Action> = EpsilonGreedyPolicy(
    q = q,
    stateActions = stateActions,
    epsilon = epsilon,
    rng = rng)

fun <State, Action> softMaxPolicy(
    q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    temperature: ParameterSchedule,
    rng: Random = Random.Default
): SoftmaxPolicy<State, Action> = SoftmaxPolicy(
    q = q,
    stateActions = stateActions,
    temperature = temperature,
    rng = rng
)

fun <State, Action> epsilonSoftPolicy(
    q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): EpsilonSoftPolicy<State, Action> = EpsilonSoftPolicy(
    q = q,
    stateActions = stateActions,
    epsilon = epsilon,
    rng = rng
)

fun <State, Action> uniformRandomPolicy(
    q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,

    ): StochasticPolicy<State, Action> = UniformStochasticPolicy(q, stateActions)

fun constantParameterSchedule(value: Double) = ParameterSchedule { value }

fun linearDecaySchedule(
    initialValue: Double,
    decayRate: Double,
    minValue: Double,
    burnInEpisodes: Int = 0,
    callback: (Int, Double) -> Unit = { _, _ -> }
): Pair<ParameterSchedule, ParameterScheduleDecay> {

    var episode = 0
    var parameter = initialValue

    val schedule = ParameterSchedule {
        parameter
    }

    val decrement: ParameterScheduleDecay = {
        if (episode >= burnInEpisodes) {
            parameter = (parameter - decayRate).coerceAtLeast(minValue)
        }
        episode++
        callback(episode, parameter)
    }

    return schedule to decrement
}

typealias ParameterScheduleDecay = () -> Unit