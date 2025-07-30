package io.github.kotlinrl.core

import kotlin.random.*

typealias ParameterSchedule = io.github.kotlinrl.core.policy.ParameterSchedule
typealias RandomPolicy<State, Action> = io.github.kotlinrl.core.policy.RandomPolicy<State, Action>
typealias GreedyPolicy<State, Action> = io.github.kotlinrl.core.policy.GreedyPolicy<State, Action>
typealias EpsilonGreedyPolicy<State, Action> = io.github.kotlinrl.core.policy.EpsilonGreedyPolicy<State, Action>
typealias SoftmaxPolicy<State, Action> = io.github.kotlinrl.core.policy.SoftmaxPolicy<State, Action>
typealias EpsilonSoftPolicy<State, Action> = io.github.kotlinrl.core.policy.EpsilonSoftPolicy<State, Action>
typealias Policy<State, Action> = io.github.kotlinrl.core.policy.Policy<State, Action>
typealias StochasticPolicy<State, Action> = io.github.kotlinrl.core.policy.StochasticPolicy<State, Action>
typealias PolicyProbabilities<State, Action> = io.github.kotlinrl.core.policy.PolicyProbabilities<State, Action>
typealias StateActionListProvider<State, Action> = io.github.kotlinrl.core.policy.StateActionListProvider<State, Action>
typealias MutablePolicy<State, Action> = io.github.kotlinrl.core.policy.MutablePolicy<State, Action>
typealias ProbabilityFunction<State, Action> = io.github.kotlinrl.core.policy.ProbabilityFunction<State, Action>
typealias QFunctionPolicy<State, Action> = io.github.kotlinrl.core.policy.QFunctionPolicy<State, Action>

fun <State, Action> randomPolicy(
    actionProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(actionProvider, rng)

fun <State, Action> greedyPolicy(
    qTable: QFunction<State, Action>
): QFunctionPolicy<State, Action> = GreedyPolicy(qTable)

fun <State, Action> epsilonGreedyPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    epsilon: ParameterSchedule,
    qTable: QFunction<State, Action>,
    rng: Random = Random.Default
): QFunctionPolicy<State, Action> = EpsilonGreedyPolicy(stateActionListProvider, qTable, epsilon, rng)

fun <State, Action> softMaxPolicy(
    qTable: QFunction<State, Action>,
    temperature: ParameterSchedule,
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): StochasticPolicy<State, Action> = SoftmaxPolicy(
    qTable = qTable,
    temperature = temperature,
    stateActionListProvider = stateActionListProvider,
    rng = rng
)

fun <State, Action> epsilonSoftPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>,
    qTable: QFunction<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): StochasticPolicy<State, Action> = EpsilonSoftPolicy(
    stateActionListProvider = stateActionListProvider,
    qTable = qTable,
    epsilon = epsilon,
    rng = rng
)

fun <State, Action> ProbabilityFunction<State, Action>.asPolicyProbabilities(
    stateActionListProvider: StateActionListProvider<State, Action>
): PolicyProbabilities<State, Action> = PolicyProbabilities { state ->
    stateActionListProvider(state).associateWith { action -> this.invoke(state, action) }
}

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