package io.github.kotlinrl.tabular

/**
 * A type alias for `io.github.kotlinrl.tabular.dp.PolicyIteration`.
 *
 * This alias represents the Policy Iteration algorithm, a dynamic programming method
 * used to solve Markov Decision Processes (MDPs). The algorithm alternates between
 * evaluating a policy's value function (Policy Evaluation) and improving the policy
 * based on the evaluated value function (Policy Improvement) until convergence to the
 * optimal policy and value function.
 *
 * Using this alias simplifies the reference to the implementation of the Policy Iteration
 * algorithm in the library.
 */
typealias PolicyIteration = io.github.kotlinrl.tabular.dp.PolicyIteration
/**
 * A type alias for the ValueIteration class.
 *
 * The Value Iteration algorithm is a dynamic programming method for solving Markov
 * Decision Processes (MDPs). It computes the optimal policy and value function for
 * a given MDP by iteratively applying the Bellman optimality equation until convergence.
 *
 * This alias provides a simplified reference to the corresponding implementation.
 */
typealias ValueIteration = io.github.kotlinrl.tabular.dp.ValueIteration
/**
 * A type alias for the `IncrementalMonteCarloControl` class, which implements the Incremental Monte Carlo Control
 * algorithm used in reinforcement learning. This algorithm determines optimal policies and estimates action-value
 * functions through Monte Carlo updates, supporting both first-visit and every-visit control strategies.
 *
 * This alias provides a more accessible and convenient way to reference the full class name from the library.
 */
typealias IncrementalMonteCarloControl = io.github.kotlinrl.tabular.mc.IncrementalMonteCarloControl
/**
 * Type alias for `OffPolicyMonteCarloControl`, representing the Off-Policy Monte Carlo
 * control algorithm for reinforcement learning.
 *
 * This alias refers to an implementation of a reinforcement learning algorithm that uses
 * importance sampling to estimate returns for state-action pairs and iteratively improves
 * the target policy based on collected trajectories of agent-environment interactions.
 *
 * Useful for scenarios requiring off-policy learning with Monte Carlo methods, particularly
 * in cases where an epsilon-greedy policy or other adaptive exploration strategies are employed.
 */
typealias OffPolicyMonteCarloControl = io.github.kotlinrl.tabular.mc.OffPolicyMonteCarloControl
/**
 * Type alias for the OnPolicyMonteCarloControl class from the KotlinRL library.
 *
 * This alias simplifies access to the implementation of the On-Policy Monte Carlo Control
 * algorithm, a technique used in reinforcement learning to estimate optimal policies and
 * action-value functions (Q-values) based on complete episodic experiences.
 *
 * The algorithm leverages an epsilon-greedy exploration strategy to balance exploration
 * and exploitation during learning. It processes trajectories to improve the policy
 * governing action selection and supports configurations for handling state-action
 * pairs on either every visit or first visit only.
 */
typealias OnPolicyMonteCarloControl = io.github.kotlinrl.tabular.mc.OnPolicyMonteCarloControl
/**
 * Defines an alias for the ExpectedSARSA class from the KotlinRL library, which implements the
 * Expected SARSA algorithm for reinforcement learning.
 *
 * Expected SARSA is an on-policy temporal difference learning method that improves upon the
 * SARSA algorithm by calculating the expected Q-value for the next state under the policy's
 * action distribution. This approach offers reduced variance and improved stability during
 * training compared to standard SARSA.
 *
 * This alias simplifies references to the ExpectedSARSA class within this project.
 */
typealias ExpectedSARSA = io.github.kotlinrl.tabular.td.classic.ExpectedSARSA
/**
 * A type alias for simplifying access to the SARSA (State-Action-Reward-State-Action) algorithm class.
 *
 * SARSA is an on-policy temporal difference control method in reinforcement learning.
 * It updates Q-values based on experienced transitions and actions determined by the current policy.
 * The alias provides a concise reference to the implementation located in the specified package.
 */
typealias SARSA = io.github.kotlinrl.tabular.td.classic.SARSA
/**
 * Type alias representing Q-Learning, an off-policy temporal-difference reinforcement
 * learning algorithm for learning an optimal policy and action-value function.
 *
 * This alias is intended to simplify references to the Q-Learning implementation
 * within the library.
 */
typealias QLearning = io.github.kotlinrl.tabular.td.classic.QLearning
/**
 * Type alias for the TDZero class implementing the TD(0) algorithm.
 *
 * TDZero is a model-free reinforcement learning algorithm that incrementally
 * updates the state-value function (V) using one-step temporal differences
 * derived from observed transitions. It is a policy evaluation method capable
 * of operating in dynamic and stochastic environments by leveraging feedback
 * from the observed dynamics of the process.
 *
 * This alias simplifies usage and improves code readability when working
 * with tabular TD(0) learning in the library.
 */
typealias TDZero = io.github.kotlinrl.tabular.td.classic.TDZero
/**
 * Applies the Policy Iteration algorithm to compute the optimal policy for a given environment.
 *
 * The Policy Iteration algorithm alternates between policy evaluation and policy improvement steps
 * to find the optimal policy and corresponding value function for a Markov Decision Process (MDP).
 * Policy evaluation computes the value function for the current policy, while policy improvement
 * updates the policy based on the computed value function. The process is repeated until convergence.
 *
 * @param theta The convergence threshold for the policy iteration process. The algorithm
 *              stops when the difference between consecutive policies is less than this value.
 *              Defaults to 1e-6.
 * @return A PolicyPlanner instance configured to perform policy iteration with the given
 *         convergence threshold.
 */
fun policyIteration(
    theta: Double = 1e-6,
): PolicyPlanner = PolicyIteration(theta)

/**
 * Creates an instance of the Value Iteration algorithm with a specified convergence threshold.
 *
 * Value Iteration is a dynamic programming algorithm used to compute an optimal policy
 * and value function for a given Markov Decision Process (MDP). The algorithm iteratively updates
 * the value function for states until the change in values is less than the specified threshold (theta).
 *
 * @param theta the convergence threshold for value updates. The algorithm stops iterating
 *              when the maximum difference between successive value updates is less than this value.
 *              Default value is 1e-6.
 * @return a PolicyPlanner implementation based on the Value Iteration algorithm to solve MDPs.
 */
fun valueIteration(
    theta: Double = 1e-6
): PolicyPlanner = ValueIteration(theta)