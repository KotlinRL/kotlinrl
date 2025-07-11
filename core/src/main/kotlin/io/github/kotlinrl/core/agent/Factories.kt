package io.github.kotlinrl.core.agent

import io.github.kotlinrl.core.learn.QTable
import io.github.kotlinrl.core.learn.tabular.ExpectedSARSA
import io.github.kotlinrl.core.learn.tabular.QLearning
import io.github.kotlinrl.core.learn.tabular.SARSA
import io.github.kotlinrl.core.policy.*
import java.util.*

fun <State, Action> agent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onExperience: ExperienceObserver<State, Action> = ExperienceObserver{ }
): Agent<State, Action> = BasicAgent(id, policy, onExperience)

fun qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double
): Agent<IntArray, Int>  = agent(id, policy, QLearning(
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
    val learning = SARSA(
        qTable = qTable,
        alpha = alpha,
        gamma = gamma
    )
    return agent(id, policy, learning).withStateActionCallback(learning)
}

fun expectedSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    policyProbabilities: PolicyProbabilities<IntArray, Int>
): Agent<IntArray, Int> = agent(id, policy, ExpectedSARSA(
    qTable = qTable,
    alpha = alpha,
    gamma = gamma,
    stateActionListProvider = stateActionListProvider,
    policyProbabilities = policyProbabilities
))

fun monteCarloAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>
): Agent<IntArray, Int> {
    return agent(id = id, policy = policy) { /* no-op TransitionObserver */ }
}

fun <State, Action> Agent<State, Action>.withStateActionCallback(
    callback: StateActionCallback<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun act(state: State): Action {
        callback.before(state)
        val action = this@withStateActionCallback.act(state)
        callback.after(state, action)
        return action
    }
}

fun <State, Action> Agent<State, Action>.withExperienceCallback(
    callback: ExperienceCallback<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun observe(experience: Experience<State, Action>) {
        callback.before()
        this@withExperienceCallback.observe(experience)
        callback.after(experience)
    }
}
