package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*

/**
 * A wrapper class for an environment that applies a custom reward transformation function
 * to the reward values obtained from the environment's step results.
 *
 * This class extends the functionality of a simple environment wrapper by modifying
 * the reward component of the `StepResult` using a user-defined transformation function.
 * The transformation function is applied to the original `StepResult` returned by the
 * wrapped environment, and a new `StepResult` with the transformed reward is returned.
 *
 * @param State The type representing the state of the environment.
 * @param Action The type representing actions that can be performed in the environment.
 * @param ObservationSpace The type of space defining the observation space of the environment.
 * @param ActionSpace The type of space defining the action space of the environment.
 * @param env An instance of the environment being wrapped.
 *            This environment provides the original `StepResult` values.
 * @param transform A function that takes a `StepResult` as input and returns
 *                  a transformed reward value as a `Double`.
 */
class TransformReward<State, Action, ObservationSpace : Space<State>, ActionSpace : Space<Action>>(
    env: Env<State, Action, ObservationSpace, ActionSpace>,
    private val transform: (StepResult<State>) -> Double
) : SimpleWrapper<State, Action, ObservationSpace, ActionSpace>(env) {

    /**
     * Executes a single step in the environment with the specified action and
     * applies a custom transformation to the reward.
     *
     * This method first calls the `step` method of the wrapped environment, obtaining
     * a `StepResult` that describes the outcome of the step. The reward value in the
     * `StepResult` is then transformed using the user-defined transformation function,
     * and a new `StepResult` with the updated reward is returned.
     *
     * @param action The action to be performed in the environment.
     * @return A `StepResult` containing the updated state, transformed reward,
     * termination status, truncation status, and additional information after the action is applied.
     */
    override fun step(action: Action): StepResult<State> {
        val stepResult = env.step(action)
        return stepResult.copy(reward = transform(stepResult))
    }
}
