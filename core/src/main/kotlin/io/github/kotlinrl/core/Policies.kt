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
 * Creates and returns a random policy based on the given Q-function and state-action space.
 * This policy selects actions stochastically, providing an exploration mechanism
 * or baseline behavior in reinforcement learning contexts.
 *
 * @param Q The Q-function representing the state-action value estimates.
 * @param stateActions A definition of the actions available for each possible state in the environment.
 * @param rng The random number generator used for stochastic action selection.
 * @return A stochastic policy that selects actions randomly for the specified Q-function and state-action space.
 */
fun <State, Action> randomPolicy(
    Q: QFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    rng: Random = Random.Default
): Policy<State, Action> = RandomPolicy(Q, stateActions, rng)

/**
 * Creates a greedy policy that deterministically selects the action with the highest Q-value
 * for a given state, based on the provided Q-function and state-actions mapping.
 *
 * @param Q The Q-function that provides the Q-value for state-action pairs.
 * @param stateActions A mapping that defines the valid actions for each state.
 * @return A greedy policy that selects actions to maximize the Q-value.
 */
fun <State, Action> greedyPolicy(
    Q: QFunction<State, Action>,
    stateActions: StateActions<State, Action>,
): Policy<State, Action> = GreedyPolicy(Q, stateActions)

/**
 * Creates an epsilon-greedy policy for action selection in reinforcement learning.
 *
 * The epsilon-greedy policy balances exploration and exploitation by choosing a random action
 * with a probability defined by epsilon, and selecting the action with the highest Q-value otherwise.
 *
 * @param Q the Q-function used to estimate the value of state-action pairs.
 * @param stateActions a function defining the set of available actions for each state.
 * @param epsilon a parameter schedule determining the exploration rate.
 * @param rng the random number generator used to introduce randomness in action selection.
 * @return a policy implementing the epsilon-greedy action selection strategy.
 */
fun <State, Action> epsilonGreedyPolicy(
    Q: QFunction<State, Action>,
    stateActions: StateActions<State, Action>,
    epsilon: ParameterSchedule,
    rng: Random = Random.Default
): Policy<State, Action> = EpsilonGreedyPolicy(
    Q = Q,
    stateActions = stateActions,
    epsilon = epsilon,
    rng = rng
)

/**
 * Creates a softmax policy for action selection in reinforcement learning.
 *
 * The resulting policy assigns probabilities to each possible action in a given state,
 * based on the Q-values of the state-action pairs and a temperature parameter. The
 * temperature parameter controls the trade-off between exploration and exploitation.
 *
 * @param State the type representing the state in the environment.
 * @param Action the type representing the actions available to the agent.
 * @param Q the Q-function mapping state-action pairs to their utility or quality values.
 * @param stateActions a function to determine the set of actions available for a given state.
 * @param temperature a parameter schedule controlling the temperature used in the softmax
 *        computation, which affects the stochasticity of action selection.
 * @param rng a random number generator to introduce randomness into the action selection process.
 * @return an instance of `SoftmaxPolicy` representing the softmax-based stochastic policy.
 */
fun <State, Action> softMaxPolicy(
    Q: QFunction<State, Action>,
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
 * Constructs an epsilon-soft policy using the provided Q-function, state-action mapping,
 * epsilon parameter schedule, and random number generator. The epsilon-soft policy
 * promotes exploration by occasionally selecting non-greedy actions with a probability
 * controlled by epsilon, while favoring the optimal actions determined by the Q-function.
 *
 * @param State the type representing the states in the environment.
 * @param Action the type representing the actions in the environment.
 * @param Q the Q-function that estimates the value of state-action pairs.
 * @param stateActions a functional interface that provides the available actions for a given state.
 * @param epsilon a parameter schedule defining the exploration rate at each point in time.
 * @param rng a random number generator for introducing stochasticity in action selection, defaulting to Random.Default.
 * @return an instance of an epsilon-soft policy, balancing exploration and exploitation in decision-making.
 */
fun <State, Action> epsilonSoftPolicy(
    Q: QFunction<State, Action>,
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
 * Generates a uniform random policy for a given Q-function and state-action pairings.
 * The resulting policy chooses actions uniformly at random from the available actions
 * for each state, without consideration of Q-function values.
 *
 * @param Q the Q-function that represents the expected rewards for state-action pairs.
 * @param stateActions a mapping or accessor that provides all available actions for a given state.
 * @return a stochastic policy that selects actions uniformly at random for a given state.
 */
fun <State, Action> uniformRandomPolicy(
    Q: QFunction<State, Action>,
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