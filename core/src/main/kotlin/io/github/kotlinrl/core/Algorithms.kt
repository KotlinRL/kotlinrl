package io.github.kotlinrl.core

import io.github.kotlinrl.core.algorithms.dp.PolicyIteration
import io.github.kotlinrl.core.algorithms.dp.ValueIteration
import io.github.kotlinrl.core.algorithms.td.TabularTDLearning
import io.github.kotlinrl.core.env.StepResult

typealias QFunction<State, Action> = io.github.kotlinrl.core.algorithms.QFunction<State, Action>
typealias QTable = io.github.kotlinrl.core.algorithms.QTable
typealias VTable = io.github.kotlinrl.core.algorithms.VTable
typealias PTable = io.github.kotlinrl.core.algorithms.PTable
typealias ValueIteration<State, Action> = ValueIteration<State, Action>
typealias PolicyIteration<State, Action> = PolicyIteration<State, Action>
typealias OnPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OnPolicyMonteCarloControl<State, Action>
typealias ConstantAlphaMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.ConstantAlphaMonteCarloControl<State, Action>
typealias OffPolicyMonteCarloControl<State, Action> = io.github.kotlinrl.core.algorithms.mc.OffPolicyMonteCarloControl<State, Action>
typealias ExpectedSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.ExpectedSARSA<State, Action>
typealias QLearning<State, Action> = io.github.kotlinrl.core.algorithms.td.QLearning<State, Action>
typealias SARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.SARSA<State, Action>
typealias NStepSARSA<State, Action> = io.github.kotlinrl.core.algorithms.td.nstep.NStepSARSA<State, Action>
typealias TabularTDLearning<State, Action> = TabularTDLearning<State, Action>

fun <State, Action> qLearning(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double
): TransitionObserver<State, Action> = QLearning(qTable, alpha, gamma)

fun <State, Action> sarsa(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double
): TransitionObserver<State, Action> = SARSA(qTable, alpha, gamma)

fun <State, Action> expectedSARSA(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<State, Action>,
    policyProbabilities: PolicyProbabilities<State, Action>
): TransitionObserver<State, Action> = ExpectedSARSA(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma,
    stateActionListProvider = stateActionListProvider,
    policyProbabilities = policyProbabilities
)

fun <State, Action> nStepSARSA(
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double,
    n: Int,
    policyProbabilities: PolicyProbabilities<State, Action>
): NStepSARSA<State, Action> = NStepSARSA(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma,
    n = n,
    policyProbabilities = policyProbabilities
)

fun <State, Action> valueIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    stateActionListProvider: StateActionListProvider<State, Action>,
    actionComparator: Comparator<Action>
): Policy<State, Action> = valueIterationPlanner(gamma, theta, vTable, pTable, actionComparator)
    .plan(
        stateActionListProvider = stateActionListProvider,
        transitionFunction = { state, action ->
            val stepResult: StepResult<State> = env.simulateStep(state, action)
            Transition(
                state = state,
                action = action,
                reward =  stepResult.reward,
                nextState = stepResult.state,
                terminated = stepResult.terminated,
                truncated = stepResult.truncated,
                info = stepResult.info
            )
        }
    )

fun <State, Action> valueIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    stateActionListProvider: StateActionListProvider<State, Action>,
    transitionFunction: TransitionFunction<State, Action>,
    actionComparator: Comparator<Action>
): Policy<State, Action> = valueIterationPlanner(gamma, theta, vTable, pTable, actionComparator)
    .plan(
        stateActionListProvider = stateActionListProvider,
        transitionFunction = transitionFunction
    )


fun <State, Action> policyIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    stateActionListProvider: StateActionListProvider<State, Action>
): Policy<State, Action> = policyIterationPlanner(gamma, theta, vTable, pTable)
    .plan(
        stateActionListProvider = stateActionListProvider,
        transitionFunction = { state, action ->
            val stepResult: StepResult<State> = env.simulateStep(state, action)
            Transition(
                state = state,
                action = action,
                reward =  stepResult.reward,
                nextState = stepResult.state,
                terminated = stepResult.terminated,
                truncated = stepResult.truncated,
                info = stepResult.info
            )
        }
    )

fun policyIteration(
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: VTable,
    pTable: PTable,
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    transitionFunction: TransitionFunction<IntArray, Int>
): Policy<IntArray, Int> = policyIterationPlanner(gamma, theta, vTable, pTable)
    .plan(
        stateActionListProvider = stateActionListProvider,
        transitionFunction = transitionFunction
    )

fun onPolicyMonteCarloControl(
    qTable: QTable,
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true
): EpisodeCallback<IntArray, Int> = OnPolicyMonteCarloControl(qTable, gamma, firstVisitOnly)

fun constantAlphaMonteCarloControl(
    qTable: QTable,
    gamma: Double = 0.99,
    alpha: Double = 0.05,
    firstVisitOnly: Boolean = true
): EpisodeCallback<IntArray, Int> = ConstantAlphaMonteCarloControl(qTable, gamma, alpha, firstVisitOnly)

fun offPolicyMonteCarloControl(
    qTable: QTable,
    gamma: Double = 0.99,
    behaviorPolicy: ProbabilisticPolicy<IntArray, Int>,
    targetPolicy: MutablePolicy<IntArray, Int>
): EpisodeCallback<IntArray, Int> = OffPolicyMonteCarloControl(
    qTable = qTable,
    gamma = gamma,
    behaviorPolicy = behaviorPolicy,
    targetPolicy = targetPolicy
)