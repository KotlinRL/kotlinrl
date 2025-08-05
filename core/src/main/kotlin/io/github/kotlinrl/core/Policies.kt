package io.github.kotlinrl.core

import kotlin.random.*

/**
 * A type alias for `io.github.kotlinrl.core.policy.ParameterSchedule`.
 *
 * Represents a schedule to dynamically adjust a parameter value in reinforcement
 * learning algorithms. This schedule is typically a function that provides the
 * current value of a parameter based on some dynamic condition, such as time or
 * iteration count.
 *
 * Commonly used for parameters like exploration rates or temperature in policies
 * where the value needs adjustment over time.
 */
typealias ParameterSchedule = io.github.kotlinrl.core.policy.ParameterSchedule
/**
 * A type alias for `io.github.kotlinrl.core.policy.RandomPolicy`.
 *
 * Represents a stochastic policy where actions are selected randomly from the
 * action space of a given state. This policy is commonly used for baseline
 * performance comparison or exploration in reinforcement learning.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias RandomPolicy<State, Action> = io.github.kotlinrl.core.policy.RandomPolicy<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.policy.GreedyPolicy`.
 *
 * Represents a deterministic policy that selects the action with the highest Q-value
 * for a given state. The policy leverages a provided Q-function and state-actions mapping
 * to make greedy decisions that aim to maximize reward.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the actions that can be performed in the environment.
 */
typealias GreedyPolicy<State, Action> = io.github.kotlinrl.core.policy.GreedyPolicy<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.policy.EpsilonGreedyPolicy`.
 *
 * Represents an epsilon-greedy policy implementation for reinforcement learning. This policy enables
 * action selection by balancing exploration and exploitation strategies, governed by a parameter
 * epsilon. Within this strategy, actions are chosen randomly with a probability of epsilon,
 * while the remaining probability is assigned to selecting the greedy (highest Q-value) action.
 *
 * This type alias simplifies the reference to the `EpsilonGreedyPolicy` class, which implements this
 * behavior for selecting actions given the current state in a reinforcement learning setting.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing the actions that can be executed within the environment.
 */
typealias EpsilonGreedyPolicy<State, Action> = io.github.kotlinrl.core.policy.EpsilonGreedyPolicy<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.policy.SoftmaxPolicy`.
 *
 * This alias provides a more concise way to reference the `SoftmaxPolicy` class, which is a policy
 * used in reinforcement learning. It selects actions stochastically based on the softmax function,
 * factoring in the Q-values for state-action pairs and a temperature parameter that controls the balance
 * between exploration and exploitation.
 *
 * @param State The type representing the states in the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias SoftmaxPolicy<State, Action> = io.github.kotlinrl.core.policy.SoftmaxPolicy<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.policy.EpsilonSoftPolicy`.
 *
 * Represents a stochastic policy implementation that uses epsilon-soft action selection.
 * This policy ensures exploration of actions with a probability determined by epsilon,
 * while following a greedy policy with the complementary probability (1 - epsilon).
 * It is used in reinforcement learning environments to balance exploration and exploitation.
 *
 * @param State The type representing the environment's state.
 * @param Action The type representing the possible actions within the environment.
 */
typealias EpsilonSoftPolicy<State, Action> = io.github.kotlinrl.core.policy.EpsilonSoftPolicy<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.policy.Policy`, representing a policy in reinforcement learning.
 *
 * A policy defines the behavior of an agent by determining the action to be taken
 * based on the current state of the environment. This type alias simplifies the reference
 * to the policy interface, which accommodates deterministic or stochastic strategies for decision-making.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the actions available to the agent.
 */
typealias Policy<State, Action> = io.github.kotlinrl.core.policy.Policy<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.policy.QFunction`.
 *
 * Represents a Q-function, which estimates the value of state-action pairs in a reinforcement
 * learning context. The Q-function is a key component in reinforcement learning that defines
 * the expected cumulative reward for performing a given action in a specific state, generally
 * used for decision-making and policy optimization.
 *
 * This alias simplifies references to the interface, providing more concise type definitions
 * when working with Q-functions in the context of state-action value estimation.
 *
 * @param State The type representing the state space of the environment.
 * @param Action The type representing the action space of the environment.
 */
typealias QFunction<State, Action> = io.github.kotlinrl.core.policy.QFunction<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.policy.QFunctionPolicy`.
 *
 * Represents a policy in reinforcement learning derived from a Q-function,
 * used to determine the behavior of an agent based on the quality of state-action pairs.
 *
 * This policy provides abstraction for computing action probabilities and
 * determining the likelihood of selecting a specific action in a given state.
 * The underlying Q-function evaluates the expected cumulative reward, enabling
 * the policy to make decisions aimed at maximizing performance in an environment.
 *
 * @param State The type representing the state in the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias QFunctionPolicy<State, Action> = io.github.kotlinrl.core.policy.QFunctionPolicy<State, Action>
/**
 * Typealias for `io.github.kotlinrl.core.policy.EnumerableQFunction`, providing a more concise reference
 * to an interface that represents a Q-function with an enumerable state space.
 *
 * The `EnumerableQFunction` is a specialized version of the `QFunction` interface, designed for cases where
 * all possible states in an environment can be explicitly enumerated. It facilitates algorithms that require
 * complete knowledge and explicit iteration over the state space.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the action that can be taken in the environment.
 */
typealias EnumerableQFunction<State, Action> = io.github.kotlinrl.core.policy.EnumerableQFunction<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.policy.ValueFunction`.
 *
 * Represents a value function in reinforcement learning, mapping states to scalar values
 * representing the estimated value of those states. This simplifies the reference to the
 * `ValueFunction` interface used in algorithms to evaluate and compare states based on
 * their expected future rewards.
 *
 * @param State The type representing the state in the environment.
 */
typealias ValueFunction<State> = io.github.kotlinrl.core.policy.ValueFunction<State>
/**
 * A type alias for `io.github.kotlinrl.core.policy.Planner`.
 *
 * Represents a functional interface for creating decision-making policies
 * in reinforcement learning. A `Planner` is used to generate a policy based
 * on the current state of the environment and available actions, facilitating
 * the implementation of various planning strategies.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing the actions available in the environment.
 */
typealias Planner<State, Action> = io.github.kotlinrl.core.policy.Planner<State, Action>
/**
 * A type alias for the `io.github.kotlinrl.core.policy.EnumerableValueFunction` interface.
 *
 * Represents a value function where all possible states are enumerable, allowing explicit
 * iteration over the state space. This facilitates the implementation of algorithms that
 * require comprehensive knowledge of all states, such as those in reinforcement learning.
 *
 * Delegates to `io.github.kotlinrl.core.policy.EnumerableValueFunction`.
 *
 * @param State The type representing the states in the environment.
 */
typealias EnumerableValueFunction<State> = io.github.kotlinrl.core.policy.EnumerableValueFunction<State>
/**
 * A type alias for `io.github.kotlinrl.core.policy.StochasticPolicy`.
 *
 * Represents a stochastic policy in the context of reinforcement learning, which determines
 * actions probabilistically based on their computed scores or Q-values. This type alias
 * provides a simplified reference to the generic `StochasticPolicy` class for use within
 * the codebase.
 *
 * @param State The type representing the environment's state.
 * @param Action The type representing the possible actions in the environment.
 */
typealias StochasticPolicy<State, Action> = io.github.kotlinrl.core.policy.StochasticPolicy<State, Action>
/**
 * Type alias for `io.github.kotlinrl.core.policy.UniformStochasticPolicy`.
 *
 * Represents a stochastic policy that selects actions uniformly at random
 * from the set of available actions for a given state. This type alias provides
 * a simplified reference to the `UniformStochasticPolicy` class, which ensures
 * actions are chosen with equal probability irrespective of prior knowledge or
 * Q-function values.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions that can be performed in the environment.
 */
typealias UniformStochasticPolicy<State, Action> = io.github.kotlinrl.core.policy.UniformStochasticPolicy<State, Action>
/**
 * A type alias for `io.github.kotlinrl.core.policy.StateActions`.
 *
 * Represents a functional interface used to define the contract for determining
 * the possible actions available for a specific state in an environment.
 * It is primarily used in reinforcement learning contexts to identify the
 * action space for given states.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions available in the environment.
 */
typealias StateActions<State, Action> = io.github.kotlinrl.core.policy.StateActions<State, Action>
/**
 * Defines a type alias for a function that updates a policy in reinforcement learning.
 *
 * This function takes an instance of `Policy` and modifies it to reflect a new or updated
 * policy based on the implementation. The policy encapsulates the behavior of how actions
 * are chosen in any given state.
 *
 * @param State The type representing the state in the policy.
 * @param Action The type representing the action in the policy.
 */
typealias PolicyUpdate<State, Action> = (Policy<State, Action>) -> Unit
/**
 * Type alias for a function that updates an `EligibilityTrace`.
 *
 * Represents a functional abstraction for modifying an eligibility trace,
 * which is a mechanism used in reinforcement learning to keep track of
 * state-action pairs and their eligibility for learning updates.
 *
 * The provided function specifies how an eligibility trace should be updated
 * based on its current state.
 *
 * @param State The type representing the environment states.
 * @param Action The type representing the actions that can be taken in the environment.
 */
typealias EligibilityTraceUpdate<State, Action> = (EligibilityTrace<State, Action>) -> Unit

/**
 * Creates a random policy that selects actions based on a uniform random distribution
 * over the available actions for a given state.
 *
 * @param stateActions A mapping that defines the available actions for each state.
 * @param rng An instance of a random number generator to use. Defaults to [Random.Default].
 * @return A policy that selects actions uniformly at random for a given state.
 */
fun <State, Action> randomPolicy(
    stateActions: StateActions<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(stateActions, rng)

/**
 * Constructs a greedy policy based on the given Q-function and state-actions mapping.
 *
 * This policy selects the action with the highest Q-value for a given state,
 * ensuring deterministic action selection that aims to maximize reward.
 *
 * @param Q The Q-function to evaluate state-action pairs and compute Q-values.
 * @param stateActions The mapping of states to the available actions for each state.
 * @return A deterministic policy that follows the greedy approach to action selection.
 */
fun <State, Action> greedyPolicy(
    Q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
): QFunctionPolicy<State, Action> = GreedyPolicy(Q, stateActions)

/**
 * Creates an epsilon-greedy policy for action selection in reinforcement learning.
 * The policy selects a random action with a probability defined by the epsilon parameter
 * and selects the action with the highest Q-value otherwise.
 *
 * @param Q the Q-function used to evaluate state-action value pairs and determine the best action.
 * @param stateActions a function that provides the set of available actions for a given state.
 * @param epsilon a parameter schedule defining the exploration probability (epsilon) over time.
 * @param rng the random number generator used for selecting random actions.
 * @return an epsilon-greedy policy that balances exploration and exploitation when selecting actions.
 */
fun <State, Action> epsilonGreedyPolicy(
    Q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): QFunctionPolicy<State, Action> = EpsilonGreedyPolicy(
    Q = Q,
    stateActions = stateActions,
    epsilon = epsilon,
    rng = rng
)

/**
 * Constructs a softmax policy that selects actions stochastically based on the softmax function,
 * using Q-values for state-action pairs and a temperature parameter to balance exploration and exploitation.
 *
 * @param Q the Q-function that estimates the expected cumulative rewards for state-action pairs.
 * @param stateActions a function that retrieves the list of available actions for a given state.
 * @param temperature a parameter schedule that determines the temperature value for the softmax computation.
 *                     Higher values lead to more exploration, while lower values encourage exploitation.
 * @param rng a random number generator used for sampling actions stochastically. Defaults to the standard random generator.
 * @return an instance of SoftmaxPolicy configured with the given Q-function, state-action mapping, temperature, and random generator.
 */
fun <State, Action> softMaxPolicy(
    Q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    temperature: ParameterSchedule,
    rng: Random = Random.Default
): SoftmaxPolicy<State, Action> = SoftmaxPolicy(
    Q = Q,
    stateActions = stateActions,
    temperature = temperature,
    rng = rng
)

/**
 * Creates an epsilon-soft policy for reinforcement learning. This policy combines
 * exploration and exploitation by selecting actions stochastically based on an epsilon value.
 * The policy follows the greedy action (highest Q-value) with a probability of (1 - epsilon)
 * and explores other actions with a probability of epsilon.
 *
 * @param Q The Q-function representing the quality of state-action pairs.
 * @param stateActions A function that returns the set of available actions for a given state.
 * @param epsilon The parameter schedule defining the exploration rate (epsilon) over time.
 * @param rng The random number generator used for stochastic decisions. Default is `Random.Default`.
 * @return An `EpsilonSoftPolicy` instance utilizing the provided Q-function and exploration schedule.
 */
fun <State, Action> epsilonSoftPolicy(
    Q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): EpsilonSoftPolicy<State, Action> = EpsilonSoftPolicy(
    Q = Q,
    stateActions = stateActions,
    epsilon = epsilon,
    rng = rng
)

/**
 * Creates a uniform random policy for a reinforcement learning environment.
 * The policy selects actions uniformly at random from the set of available actions
 * for the given state, ensuring equal probability for all actions.
 *
 * @param Q the Q-function representing the expected utility of state-action pairs.
 * @param stateActions a mapping of states to their respective available actions.
 * @return a stochastic policy that chooses actions uniformly at random.
 */
fun <State, Action> uniformRandomPolicy(
    Q: EnumerableQFunction<State, Action>,
    stateActions: StateActions<State, Action>,
): StochasticPolicy<State, Action> = UniformStochasticPolicy(Q, stateActions)

/**
 * Creates a constant parameter schedule that always returns the specified value.
 *
 * This function is commonly used to maintain a parameter at a fixed value throughout
 * a reinforcement learning algorithm or any process that uses parameter schedules.
 *
 * @param value The constant parameter value to be returned by the schedule.
 * @return A `ParameterSchedule` instance that always evaluates to the given value.
 */
fun constantParameterSchedule(value: Double) = ParameterSchedule { value }

/**
 * Creates a linear decay schedule for a given parameter with optional burn-in episodes and a callback for updates.
 *
 * The linear decay schedule gradually decreases the parameter value from the `initialValue` by
 * `decayRate` on each episode until the `minValue` is reached. The decay does not start until
 * the specified `burnInEpisodes` are completed. A callback function is invoked on each step
 * to provide the current episode and parameter value.
 *
 * @param initialValue The initial value of the parameter to be scheduled.
 * @param decayRate The fixed decrement amount to subtract from the parameter value per episode.
 * @param minValue The minimum value that the parameter can decay to.
 * @param burnInEpisodes The number of episodes before the decay begins. Defaults to 0.
 * @param callback A function invoked on each episode with the current episode number and parameter value. Defaults to no operation.
 * @return A pair containing the parameter schedule (`ParameterSchedule`) and its associated decay function (`ParameterScheduleDecay`).
 */
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

/**
 * A type alias for a function that specifies the behavior of a parameter schedule decay mechanism.
 *
 * This alias represents a lambda or function that performs operations related to the
 * decay or adjustment of parameters, commonly used in scenarios such as learning rate schedules,
 * exploration-exploitation balances, or other time-based adjustments in machine learning or simulation.
 */
typealias ParameterScheduleDecay = () -> Unit