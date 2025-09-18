package io.github.kotlinrl.core.wrapper

import io.github.kotlinrl.core.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*

/**
 * A wrapper for an existing environment that flattens the observations into a one-dimensional
 * array (NDArray) format. This transformation simplifies environments with nested or
 * multidimensional observation structures, making them compatible with simpler observation
 * spaces like `Box`.
 *
 * @param Num The numeric type for the observation elements (e.g., Double, Float, Int, Long).
 * @param WrappedState The original, untransformed state type of the wrapped environment.
 * @param Action The type of actions supported by the environment.
 * @param WrappedObservationSpace The original observation space of the wrapped environment.
 * @param ActionSpace The action space defining allowable actions in the environment.
 * @param env The original environment to be wrapped and transformed.
 * @param dtype The data type to be used for the flattened observation NDArray
 *              (e.g., DoubleDataType, FloatDataType).
 */
class FlattenObservation<Num : Number, WrappedState, Action, WrappedObservationSpace : Space<WrappedState>, ActionSpace : Space<Action>>(
    env: Env<WrappedState, Action, WrappedObservationSpace, ActionSpace>,
    private val dtype: DataType
) : Wrapper<NDArray<Num, D1>, Action, Box<Num, D1>, ActionSpace, WrappedState, Action, WrappedObservationSpace, ActionSpace>(
    env
) {

    /**
     * Represents the observed space of the environment in a flattened format.
     *
     * This property defines the bounds of the observation space, considering the data type specified by `dtype`.
     * It utilizes a lazy initialization approach to compute the appropriate lower and upper bounds for the
     * flattened observation space, ensuring compliance with the data type and observation dimensionality.
     *
     * The observation space is represented as a `Box`, where:
     * - `low` specifies the minimum values for each dimension.
     * - `high` specifies the maximum values for each dimension.
     * - `dtype` determines the numerical type used (e.g., Double, Float, Int, Long).
     */
    override val observationSpace: Box<Num, D1> by lazy {
        val sampleObs = env.observationSpace.sample()
        val flatList = flattenObservation(sampleObs, dtype)
        val flatLength = flatList.size

        @Suppress("UNCHECKED_CAST")
        val low = when (dtype) {
            DoubleDataType -> mk.ndarray<Double, D1>(
                List(flatLength) { Double.NEGATIVE_INFINITY },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            FloatDataType -> mk.ndarray<Float, D1>(
                List(flatLength) { Float.NEGATIVE_INFINITY },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            IntDataType -> mk.ndarray<Int, D1>(
                List(flatLength) { Int.MIN_VALUE },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            LongDataType -> mk.ndarray<Long, D1>(
                List(flatLength) { Long.MIN_VALUE },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            else -> throw IllegalArgumentException("Unsupported dtype: $dtype")
        }

        @Suppress("UNCHECKED_CAST")
        val high = when (dtype) {
            DoubleDataType -> mk.ndarray<Double, D1>(
                List(flatLength) { Double.POSITIVE_INFINITY },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            FloatDataType -> mk.ndarray<Float, D1>(
                List(flatLength) { Float.POSITIVE_INFINITY },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            IntDataType -> mk.ndarray<Int, D1>(
                List(flatLength) { Int.MAX_VALUE },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            LongDataType -> mk.ndarray<Long, D1>(
                List(flatLength) { Long.MAX_VALUE },
                intArrayOf(flatLength)
            ) as NDArray<Num, D1>

            else -> throw IllegalArgumentException("Unsupported dtype: $dtype")
        }
        Box(low, high, dtype)
    }

    /**
     * The action space of the environment, represented as an `ActionSpace` instance.
     * Defines the set of possible actions that can be performed in the environment.
     * This property is overridden to provide the action space of the wrapped environment.
     */
    override val actionSpace: ActionSpace
        get() = env.actionSpace

    /**
     * Resets the environment with a flattened representation of the initial state.
     *
     * This method reinitializes the environment and transforms the observation into
     * a flat `NDArray` representation. It optionally supports a random seed and additional
     * configuration options for customizing the reset process.
     *
     * @param seed An optional random seed for deterministic behavior during reset.
     *             If `null`, the default random generator is used.
     * @param options An optional map for additional configuration options that influence
     *                the reset behavior, depending on the environment's requirements.
     * @return The initial state of the environment after reset, wrapped in an `InitialState`
     *         object containing the flattened state as an `NDArray` and additional metadata.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<NDArray<Num, D1>> {
        val initial = env.reset(seed, options)
        val flat = toNDArray<Num>(flattenObservation(initial.state, dtype), dtype)
        return InitialState(
            state = flat,
            info = initial.info
        )
    }

    /**
     * Executes a step in the environment using the given action and returns the updated environment state as a flattened NDArray,
     * along with other step result details such as reward, termination status, truncation status, and additional info.
     *
     * @param action The action to be performed in the environment.
     * @return A StepResult containing the flattened state as an NDArray, reward, termination status, truncation status, and additional info.
     */
    override fun step(action: Action): StepResult<NDArray<Num, D1>> {
        val t = env.step(action)
        val flat = toNDArray<Num>(flattenObservation(t.state, dtype), dtype)
        return StepResult(
            state = flat,
            reward = t.reward,
            terminated = t.terminated,
            truncated = t.truncated,
            info = t.info
        )
    }
}
