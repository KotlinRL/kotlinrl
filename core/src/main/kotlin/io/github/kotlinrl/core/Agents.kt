package io.github.kotlinrl.core

import io.github.kotlinrl.core.algorithms.mc.defaultKeyFunction
import io.github.kotlinrl.core.policy.QFunctionPolicy
import java.util.*

typealias Agent<State, Action> = io.github.kotlinrl.core.agent.Agent<State, Action>
typealias ObserveTransition<State, Action> = io.github.kotlinrl.core.agent.ObserveTransition<State, Action>
typealias ObserveTrajectory<State, Action> = io.github.kotlinrl.core.agent.ObserveTrajectory<State, Action>
typealias Transition<State, Action> = io.github.kotlinrl.core.agent.Transition<State, Action>
typealias PolicyAgent<State, Action> = io.github.kotlinrl.core.agent.PolicyAgent<State, Action>
typealias TransitionLearner<S, A> = ObserveTransition<S, A>
typealias TrajectoryLearner<S, A> = ObserveTrajectory<S, A>

fun <State, Action> agent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    onTransition: ObserveTransition<State, Action> = ObserveTransition { },
    onTrajectory: ObserveTrajectory<State, Action> = ObserveTrajectory { _, _ -> }
): Agent<State, Action> = PolicyAgent(
    id = id,
    policy = policy,
    onTransition = onTransition,
    onTrajectory = onTrajectory,
)

fun <State, Action> valueIterationAgent(
    id: String = UUID.randomUUID().toString(),
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    stateActionListProvider: StateActionListProvider<State, Action>,
    actionComparator: Comparator<Action>
): Agent<State, Action> = agent(
    id = id,
    policy = valueIteration(
        gamma = gamma,
        theta = theta,
        env = env,
        pTable = pTable,
        vTable = vTable,
        stateActionListProvider = stateActionListProvider,
        actionComparator = actionComparator
    )
)

fun <State, Action> policyIterationAgent(
    id: String = UUID.randomUUID().toString(),
    gamma: Double = 0.99,
    theta: Double = 1e-6,
    vTable: ValueFunction<State>,
    pTable: MutablePolicy<State, Action>,
    env: ModelBasedEnv<State, Action, *, *>,
    stateActionListProvider: StateActionListProvider<State, Action>
): Agent<State, Action> = agent(
    id = id,
    policy = policyIteration(
        gamma = gamma,
        theta = theta,
        vTable = vTable,
        pTable = pTable,
        env = env,
        stateActionListProvider = stateActionListProvider
    )
)

fun <State, Action> onPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    policy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    firstVisitOnly: Boolean = true,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultKeyFunction
): Agent<State, Action> = agent(
    id = id,
    policy = policy,
    onTrajectory = onPolicyMonteCarloControl(
        qTable = policy.qTable,
        gamma = gamma,
        firstVisitOnly = firstVisitOnly,
        stateActionKeyFunction = stateActionKeyFunction
    )
)

fun <State, Action> offPolicyMonteCarloControlAgent(
    id: String = UUID.randomUUID().toString(),
    policy: QFunctionPolicy<State, Action>,
    gamma: Double = 0.99,
    probability: ProbabilityFunction<State, Action>,
    stateActionKeyFunction: StateActionKeyFunction<State, Action> = ::defaultKeyFunction
): Agent<State, Action> = agent(
    id = id,
    policy = policy,
    onTrajectory = offPolicyMonteCarloControl(
        gamma = gamma,
        targetPolicy = policy,
        probability = probability,
        stateActionKeyFunction = stateActionKeyFunction,
    )
)

fun <State, Action> qLearningAgent(
    id: String = UUID.randomUUID().toString(),
    policy: QFunctionPolicy<State, Action>,
    alpha: Double,
    gamma: Double
): Agent<State, Action> = agent(
    id = id,
    policy = policy,
    onTransition = qLearning(
        qTable = policy.qTable,
        alpha = alpha,
        gamma = gamma
    )
)

fun <State, Action> sarsaAgent(
    id: String = UUID.randomUUID().toString(),
    policy: Policy<State, Action>,
    qTable: QFunction<State, Action>,
    alpha: Double,
    gamma: Double
): Agent<State, Action> = agent(
    id = id,
    policy = policy,
    onTransition = sarsa(
        qTable = qTable,
        alpha = alpha,
        gamma = gamma
    )
)

fun <State, Action> expectedSARSAAgent(
    id: String = UUID.randomUUID().toString(),
    policy: StochasticPolicy<State, Action>,
    alpha: Double,
    gamma: Double,
): Agent<State, Action> = agent(
    id = id,
    policy = policy,
    onTransition = expectedSARSA(
        qTable = policy.qTable,
        alpha = alpha,
        gamma = gamma,
        stateActionListProvider = policy.stateActionListProvider,
        policyProbabilities = policy.asPolicyProbabilities(policy.stateActionListProvider)
    )
)
//
//fun nStepSARSAAgent(
//    id: String = UUID.randomUUID().toString(),
//    policy: StochasticPolicy<IntArray, Int>,
//    qTable: QTable,
//    alpha: Double,
//    gamma: Double,
//    n: Int,
//    stateActionListProvider: StateActionListProvider<IntArray, Int>
//): Pair<Agent<IntArray, Int>, EpisodeCallback<IntArray, Int>> {
//    val learning = nStepSARSA(
//        qTable = qTable,
//        alpha = alpha,
//        gamma = gamma,
//        n = n,
//        policyProbabilities = policy.asPolicyProbabilities(stateActionListProvider)
//    )
//    return agent(id, policy, learning) to learning
//}
//
//fun <State, Action> offPolicyMonteCarloControlAgent(
//    id: String = UUID.randomUUID().toString(),
//    policy: Policy<State, Action>
//): Agent<State, Action> {
//    return agent(id = id, policy = policy)
//}

fun <State, Action> Agent<State, Action>.withTransitionLearner(
    onTransition: TransitionLearner<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun observe(transition: Transition<State, Action>) {
        this@withTransitionLearner.observe(transition)
        onTransition(transition)
    }
}

fun <State, Action> Agent<State, Action>.withEpisodeLearner(
    onEpisode: TrajectoryLearner<State, Action>
): Agent<State, Action> = object : Agent<State, Action> by this {
    override fun observe(trajectory: List<Transition<State, Action>>, episode: Int) {
        this@withEpisodeLearner.observe(trajectory, episode)
        onEpisode(trajectory, episode)
    }
}
