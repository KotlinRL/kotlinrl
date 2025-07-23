package io.github.kotlinrl.core

import java.util.*

typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
typealias TransitionObserver<State, Action> = io.github.kotlinrl.core.agent.TransitionObserver<State, Action>
typealias Transition<State, Action> = io.github.kotlinrl.core.agent.Transition<State, Action>
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>

fun <State, Action> agent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onTransition: TransitionObserver<State, Action> = TransitionObserver { }
): Agent<State, Action> = PolicyAgent(id, policy, onTransition)

fun qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double
): Agent<IntArray, Int> = agent(id, policy, qLearning(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma
))

fun sarsaAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double
): Agent<IntArray, Int> {
    val learning = sarsa(
        qTable = qTable,
        alpha = alpha,
        gamma = gamma
    )
    return agent(id, policy, learning)
}

fun expectedSARSAAgent(
    id: String = UUID.randomUUID().toString(),
    policy: StochasticPolicy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<IntArray, Int>
): Agent<IntArray, Int> = agent(id, policy, expectedSARSA(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma,
    stateActionListProvider = stateActionListProvider,
    policyProbabilities = policy.asPolicyProbabilities(stateActionListProvider)
))

fun nStepSARSAAgent(
    id: String = UUID.randomUUID().toString(),
    policy: StochasticPolicy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double,
    n: Int,
    stateActionListProvider: StateActionListProvider<IntArray, Int>
): Pair<Agent<IntArray, Int>, EpisodeCallback<IntArray, Int>>  {
    val learning = nStepSARSA(
        qTable = qTable,
        alpha = alpha,
        gamma = gamma,
        n = n,
        policyProbabilities = policy.asPolicyProbabilities(stateActionListProvider)
    )
    return agent(id, policy, learning) to learning
}

fun <State, Action> offPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>
): Agent<State, Action> {
    return agent(id = id, policy = policy)
}

fun <State, Action> Agent<State, Action>.withTransitionObserver(
    onTransition: TransitionObserver<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun observe(transition: Transition<State, Action>) {
        this@withTransitionObserver.observe(transition)
        onTransition(transition)
    }
}
