package io.github.kotlinrl.tabular

import io.github.kotlinrl.core.api.*
import io.github.kotlinrl.core.model.TabularMDP

/**
 * Interface defining the structure and responsibilities for a PolicyPlanner.
 *
 * A PolicyPlanner is responsible for creating a policy for a given tabular Markov Decision Process (MDP).
 * The policy represents a mapping from states to actions, defining the decision-making strategy to follow
 * in the environment described by the MDP. Additionally, the PolicyPlanner computes a value function,
 * represented as a value table (VTable), which provides the expected return for each state under the
 * generated policy.
 */
interface PolicyPlanner {
    /**
     * Generates a policy and a value table for the given tabular Markov Decision Process (MDP).
     *
     * The function computes a policy, which is a mapping from states to actions that defines
     * the decision-making strategy in the environment represented by the MDP. Additionally,
     * it calculates a value table (VTable) representing the expected return for each state
     * under the generated policy.
     *
     * @param MDP the tabular Markov Decision Process (MDP) for which the policy and value table
     * are to be generated. It contains the states, actions, reward function, transition
     * probabilities, discount factor, and terminal state information describing the environment.
     * @return a pair containing the generated policy and the value table:
     * - The first element of the pair is the `Policy` that maps states to actions.
     * - The second element is the `VTable` providing the expected value for each state.
     */
    operator fun invoke(MDP: TabularMDP): Pair<Policy<Int, Int>, VTable>
}