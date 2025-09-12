package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.model.*
import io.github.kotlinrl.core.space.*

/**
 * Represents a tabular, model-based environment with explicit state and action spaces, and
 * predictable transition dynamics. The interface combines properties of both `TabularEnv` and
 * `ModelBasedEnv` using discrete state and action spaces, allowing for manipulation and simulation
 * of the environment as a Markov Decision Process (MDP).
 */
interface TabularModelBasedEnv : TabularEnv, ModelBasedEnv<Int, Int, Discrete, Discrete> {
    /**
     * Converts the current environment into a `TabularMDP` representation based on the specified discount factor.
     * This method extracts the state and action spaces, transition probabilities, and rewards of the environment
     * to construct a `TabularMDP` object, which serves as the formal representation of a Markov Decision Process.
     *
     * @param gamma The discount factor, a value between 0 and 1, which determines the weight of future rewards
     *              relative to immediate rewards in the resulting `TabularMDP`.
     * @return A `TabularMDP` representation of the current environment.
     */
    override fun asMDP(gamma: Double): TabularMDP
}