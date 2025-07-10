package io.github.kotlinrl.core.learn.tabular

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.learn.MutableQFunction

abstract class TabularTDLearning<State, Action>(
    protected val Q: MutableQFunction<State, Action>,
    protected val alpha: Double,
    protected val gamma: Double
) : ExperienceObserver<State, Action>, StateActionCallback<State, Action> {
    protected var action: Action? = null

    override fun after(state: State, action: Action) {
        this.action = action
    }
}
