package io.github.kotlinrl.core


import java.util.*

typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
typealias TrajectoryObserver<State, Action> = io.github.kotlinrl.core.agent.TrajectoryObserver<State, Action>
typealias StepCallback<State, Action> = io.github.kotlinrl.core.agent.StepCallback<State, Action>
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>
typealias Trajectory<State, Action> = io.github.kotlinrl.core.agent.Trajectory<State, Action>

fun <State, Action> agent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onTrajectory: TrajectoryObserver<State, Action> = TrajectoryObserver { }
): Agent<State, Action> = PolicyAgent(id, policy, onTrajectory)

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

fun expectedSarsaAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<IntArray, Int>,
    qTable: QTable,
    alpha: Double,
    gamma: Double,
    stateActionListProvider: StateActionListProvider<IntArray, Int>,
    policyProbabilities: PolicyProbabilities<IntArray, Int>
): Agent<IntArray, Int> = agent(id, policy, expectedSARSA(
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


fun <State, Action> Agent<State, Action>.withStepCallback(
    callback: StepCallback<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun act(state: State): Action {
        callback.beforeStep(state)
        val action = this@withStepCallback.act(state)
        callback.afterStep(state, action)
        return action
    }
}

fun <State, Action> Agent<State, Action>.withTrajectoryObserver(
    onTrajectory: TrajectoryObserver<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun observe(trajectory: Trajectory<State, Action>) {
        this@withTrajectoryObserver.observe(trajectory)
        onTrajectory(trajectory)
    }
}
