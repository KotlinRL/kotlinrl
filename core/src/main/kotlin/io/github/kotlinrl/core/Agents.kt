package io.github.kotlinrl.core


import java.util.*

typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
typealias TrajectoryObserver<State, Action> = io.github.kotlinrl.core.agent.TrajectoryObserver<State, Action>
typealias StateActionCallback<State, Action> = io.github.kotlinrl.core.agent.StateActionCallback<State, Action>
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>
typealias TrajectoryCallback<State, Action> = io.github.kotlinrl.core.agent.TrajectoryCallback<State, Action>
typealias Trajectory<State, Action> = io.github.kotlinrl.core.agent.Trajectory<State, Action>

fun <State, Action> agent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onExperience: TrajectoryObserver<State, Action> = TrajectoryObserver { }
): Agent<State, Action> = PolicyAgent(id, policy, onExperience)

fun qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double
): Agent<IntArray, Int> = agent(id, policy, QLearning(
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

fun <State, Action> Agent<State, Action>.withTrajectoryCallback(
    callback: TrajectoryCallback<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun observe(trajectory: Trajectory<State, Action>) {
        callback.before()
        this@withTrajectoryCallback.observe(trajectory)
        callback.after(trajectory)
    }
}
