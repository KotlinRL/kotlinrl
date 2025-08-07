package io.github.kotlinrl.core.algorithms.td.nstep

import io.github.kotlinrl.core.*

/**
 * Implements n-step temporal-difference (TD) learning for reinforcement learning scenarios.
 * This algorithm uses a sequence of state-action-reward transitions to update the Q-function
 * and policy based on the agent's interactions with the environment over a fixed number of steps.
 *
 * @param State the type representing the states within the environment.
 * @param Action the type representing the actions an agent can perform within the environment.
 * @param initialPolicy the policy defining the agent's initial action-selection strategy.
 * @param n the number of steps to consider for the n-step TD learning update.
 * @param estimateQ a functional interface used to estimate the Q-function from a trajectory.
 * @param onQFunctionUpdate a callback triggered when the Q-function is updated. Default is a no-op.
 * @param onPolicyUpdate a callback triggered when the policy is updated. Default is a no-op.
 */
abstract class NStepTD<State, Action>(
    initialPolicy: Policy<State, Action>,
    private val n: Int,
    private val estimateQ: EstimateQ_fromTrajectory<State, Action>,
    onQFunctionUpdate: QFunctionUpdate<State, Action> = {},
    onPolicyUpdate: PolicyUpdate<State, Action> = {},
) : TrajectoryLearningAlgorithm<State, Action>(initialPolicy, estimateQ, onPolicyUpdate, onQFunctionUpdate) {

    private val window = ArrayDeque<Transition<State, Action>>()
    private var episode = 0
    private var tailAction: Action? = null
        set(value) {
            field = value
            when(estimateQ) {
                is NStepEstimateQ_fromTrajectory -> estimateQ.tailAction = value
            }
        }

    /**
     * Observes an entire trajectory and sets the current episode number.
     *
     * @param trajectory The trajectory consisting of sequential state-action transitions
     *                   experienced by an agent in the environment.
     * @param episode The current episode number during which the trajectory occurred.
     */
    override fun observe(trajectory: Trajectory<State, Action>, episode: Int) {
        this.episode = episode
    }

    /**
     * Observes a single state-action transition and processes it for temporal-difference learning.
     *
     * The method updates the internal window of transitions and determines whether to perform
     * n-step updates based on the current state of the window. If the transition marks the end
     * of an episode ('done'), it processes the remaining transitions in the window. Otherwise,
     * the sliding window is maintained up to the specified size (`n`).
     *
     * @param transition A state-action transition consisting of the current state, action,
     *                   next state, reward, and a flag indicating whether the episode has ended.
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
     * Handles the processing of an n-step temporal difference update by managing the sliding
     * window of transitions, extracting the action associated with the n-th step, and invoking
     * the `observe` method for further processing.
     *
     * The method removes the first transition from the window and uses the next `n` transitions
     * to invoke the parent class's `observe` method. If a valid tail action exists at the `n-th`
     * step, it is stored for processing incomplete trajectories.
     *
     * @param n the number of steps to consider for temporal difference updates. Represents the
     *          length of the trajectory segment used to compute the update.
     */
    protected fun step(n: Int) {
        tailAction = window.elementAtOrNull(n)?.action
        window.removeFirst()
        super.observe(window.take(n), episode)
    }
}