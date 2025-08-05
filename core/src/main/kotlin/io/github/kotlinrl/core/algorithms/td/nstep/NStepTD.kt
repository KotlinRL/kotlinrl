package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Abstract base class for n-step Temporal Difference (TD) reinforcement learning algorithms.
 *
 * This class extends the `TrajectoryQFunctionAlgorithm` to provide a framework for implementing
 * n-step TD learning algorithms. The primary feature of n-step TD is its reliance on trajectories of
 * n transitions to update the Q-function using a bootstrap approach. It leverages a sliding window
 * to maintain transitions and uses this information to derive Q-function updates.
 *
 * The parameter `n` specifies the number of steps in the n-step update, and the associated Q-function
 * estimator is used to apply the algorithm's specific update rules.
 *
 * This class also handles terminating episodes by ensuring that the remaining transitions in the
 * trajectory window are processed correctly using the given estimator.
 *
 * Subclasses of `NStepTD` implement concrete n-step update logic, such as SARSA, based on this abstract class.
 *
 * @param State the type representing the environment's state.
 * @param Action the type representing the actions that can be taken within the environment.
 * @param initialPolicy the initial Q-function policy used by the algorithm. This policy determines the
 *                      agent's behavior and is iteratively improved.
 * @param n the number of steps used for the n-step temporal difference update. Determines the size of
 *          the trajectory window.
 * @param estimator the trajectory-based Q-function estimator responsible for computing the Q-function
 *                  updates using n-step TD logic.
 * @param onQFunctionUpdate a callback invoked after every Q-function update. Can be used for tracking
 *                          updates or combining additional behaviors with the core update.
 * @param onPolicyUpdate a callback invoked after every policy update. Can be used to respond to policy improvements.
 */
abstract class NStepTD<State, Action>(
    initialPolicy: QFunctionPolicy<State, Action>,
    private val n: Int,
    private val estimator: TrajectoryQFunctionEstimator<State, Action>,
    onQFunctionUpdate: EnumerableQFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
) : TrajectoryQFunctionAlgorithm<State, Action>(initialPolicy, estimator, onPolicyUpdate, onQFunctionUpdate) {

    private val window = ArrayDeque<Transition<State, Action>>()
    private var episode = 0
    private var tailAction: Action? = null
        set(value) {
            field = value
            when(estimator) {
                is NStepTDQFunctionEstimator -> estimator.tailAction = value
            }
        }

    /**
     * Observes a given trajectory and updates the current episode number.
     *
     * @param trajectory The trajectory containing states, actions, and rewards to be observed.
     * @param episode The episode number related to the observed trajectory.
     */
    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        this.episode = episode
    }

    /**
     * Observes a single state-action transition and updates the window used for
     * n-step temporal-difference learning. Processes the transition based on its
     * completion status and the current size of the window.
     *
     * @param transition The state-action transition to be observed, which contains
     * the current state, action, reward, next state, and completion status.
     */
    override fun observe(transition: Transition<State, Action>) {
        window.addLast(transition)

        if (!transition.done && window.size >= n + 1) {
            step(n)
        } else if(transition.done) {
            tailAction = null
            while(window.isNotEmpty()) {
                step(n)
            }
        } else {
            while (window.size > n) window.removeFirst()
        }
    }

    /**
     * Processes an n-step window and performs required updates for temporal-difference learning.
     *
     * This method utilizes the `n` parameter to determine the action to be taken from the window,
     * updates the value of `tailAction`, modifies the window by removing the oldest state-action pair,
     * and calls the parent class's `observe` method with the updated window and current episode information.
     *
     * @param n The number of steps to process in the temporal-difference learning update.
     */
    protected fun step(n: Int) {
        tailAction = window.elementAtOrNull(n)?.action
        window.removeFirst()
        super.observe(window.take(n), episode)
    }
}