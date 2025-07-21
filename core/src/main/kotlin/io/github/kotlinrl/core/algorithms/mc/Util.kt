package io.github.kotlinrl.core.algorithms.mc

typealias StateActionKeyFunction<State, Action> = (State, Action) -> StateActionKey<*, *>

data class StateActionKey<State : Comparable<State>, Action : Comparable<Action>>(
    val state: State,
    val action: Action
) : Comparable<StateActionKey<State, Action>> {
    override fun compareTo(other: StateActionKey<State, Action>): Int {
        val stateCmp = state.compareTo(other.state)
        return if (stateCmp != 0) stateCmp else action.compareTo(other.action)
    }
}

@JvmInline
value class ComparableIntList(val data: List<Int>) : Comparable<ComparableIntList> {
    override fun compareTo(other: ComparableIntList): Int {
        return data.zip(other.data).fold(0) { acc, (a, b) ->
            if (acc != 0) acc else a.compareTo(b)
        }
    }

    override fun toString() = data.joinToString(",")
}


internal fun <State, Action> defaultKeyFunction(s: State, a: Action): StateActionKey<*, *> =
    when (s) {
        is IntArray if a is Int -> StateActionKey(ComparableIntList(s.toList()), a)
        is Comparable<*> if a is Comparable<*> -> {
            @Suppress("UNCHECKED_CAST")
            StateActionKey(s as Comparable<Any>, a as Comparable<Any>)
        }
        else -> error("State ($s) and Action ($a) must be Comparable or mappable to a comparable form.")
    }