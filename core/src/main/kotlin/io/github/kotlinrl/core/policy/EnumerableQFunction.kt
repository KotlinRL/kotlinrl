package io.github.kotlinrl.core.policy

interface EnumerableQFunction<State, Action> : QFunction<State, Action> {
    fun allStates(): List<State>
    fun allActions(state: State): List<Action>
    override fun update(
        state: State,
        action: Action,
        value: Double
    ): EnumerableQFunction<State, Action>;
}
