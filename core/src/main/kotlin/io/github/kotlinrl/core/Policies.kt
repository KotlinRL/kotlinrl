package io.github.kotlinrl.core

import kotlin.random.*

typealias ParameterSchedule = io.github.kotlinrl.core.policy.ParameterSchedule
typealias PolicyImprovementStrategy<State, Action> = io.github.kotlinrl.core.policy.PolicyImprovementStrategy<State, Action>
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
typealias PolicyProbabilities<State, Action> = io.github.kotlinrl.core.policy.PolicyProbabilities<State, Action>
typealias StateActionListProvider<State, Action> = io.github.kotlinrl.core.policy.StateActionListProvider<State, Action>
typealias ProbabilityFunction<State, Action> = io.github.kotlinrl.core.policy.ProbabilityFunction<State, Action>
typealias UniformStochasticPolicy<State, Action> = io.github.kotlinrl.core.policy.UniformStochasticPolicy<State, Action>

fun <State, Action> randomPolicy(
    actionProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(actionProvider, rng)

fun <State, Action> greedyPolicy(
    q: QFunction<State, Action>
): QFunctionPolicy<State, Action> = GreedyPolicy(q)

fun <State, Action> epsilonGreedyPolicy(
    q: QFunction<State, Action>,
    stateActionListProvider: StateActionListProvider<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): QFunctionPolicy<State, Action> = EpsilonGreedyPolicy(q, stateActionListProvider, epsilon, rng)

fun <State, Action> softMaxPolicy(
    q: QFunction<State, Action>,
    temperature: ParameterSchedule,
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): SoftmaxPolicy<State, Action> = SoftmaxPolicy(
    q = q,
    temperature = temperature,
    stateActionListProvider = stateActionListProvider,
    rng = rng
)

fun <State, Action> epsilonSoftPolicy(
    q: QFunction<State, Action>,
    epsilon: ParameterSchedule,
    stateActionListProvider: StateActionListProvider<State, Action>,
    rng: Random = Random.Default
): EpsilonSoftPolicy<State, Action> = EpsilonSoftPolicy(
    stateActionListProvider = stateActionListProvider,
    q = q,
    epsilon = epsilon,
    rng = rng
)

fun <State, Action> uniformRandomPolicy(
    stateActionListProvider: StateActionListProvider<State, Action>
): StochasticPolicy<State, Action> = UniformStochasticPolicy(stateActionListProvider)

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