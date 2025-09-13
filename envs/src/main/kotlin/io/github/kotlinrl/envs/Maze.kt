package io.github.kotlinrl.envs

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.model.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.api.*
import java.awt.*
import java.awt.image.*
import kotlin.random.*

/**
 * Represents a grid-based maze environment with the capability of simulating an agent navigating through it.
 *
 * @property exploringStarts Indicates if the agent should start in a random position. If true, the agent starts in a position
 * that is not the goal.
 * @property shapedRewards Determines if the environment uses shaped rewards. If enabled, rewards may be based on distance
 * to the goal rather than standard termination goals.
 * @property render Specifies whether the environment should render graphical output of the maze.
 * @property size Specifies the dimensions of the maze grid (size x size).
 * @property metadata A map of additional metadata about the environment.
 * @param seed Optional seed value for random number generators, ensuring reproducibility in sampling actions and observations.
 *
 * Implements the `ModelBasedEnv` interface for structured RL environments with discrete spaces.
 */
class Maze(
    val render: Boolean = true,
    override val metadata: Map<String, Any?> = emptyMap(),
    seed: Int? = null,
) : TabularModelBasedEnv {
    enum class Action(val value: Int) {
        UP(0),
        RIGHT(1),
        DOWN(2),
        LEFT(3);
    }

    private val size = 5
    private val start = Pair(0, 0)
    private val goal = Pair(size - 1, size - 1)
    private var state = start
    private val maze = createMaze(size)
    override val observationSpace = Discrete(n = size * size, start = 0, seed = seed)
    override val actionSpace = Discrete(n = 4, start = 0, seed = seed)
    override val random = seed?.let { Random(it) } ?: Random.Default

    /**
     * Companion object for providing utility methods to support the functionality of the Maze class.
     *
     * The methods defined in this companion object include utilities for modifying matrix values,
     * creating maze structures, and computing distances within a maze environment. These methods
     * are private and intended for internal use within the Maze class.
     */
    private companion object {

        /**
         * Generates a maze represented as a map, where each cell in the maze is a key,
         * and its value is a list of neighboring cells that are accessible (not blocked by walls or edges).
         *
         * The method constructs a grid-based maze, defines walls, edges, and clears connectivity between
         * cells according to the defined obstacles, resulting in the final maze structure.
         *
         * @return A map where the keys are grid positions (as lists of integers) and the values are lists of
         *         neighboring positions, representing the maze*/
        fun createMaze(size: Int): Map<Pair<Int, Int>, List<Pair<Int, Int>>> {
            val maze = IntRange(0, size).flatMap { row ->
                IntRange(0, size).map { col ->
                    (row to col) to mk[
                        row - 1 to col,   // Up
                        row + 1 to col,   // Down
                        row to col - 1,   // Left
                        row to col + 1    // Right
                    ]
                }
            }.associate { (pos, neighbors) -> pos to neighbors.toList().toMutableList() }.toMutableMap()

            // Step 2: Define the edges of the maze
            val leftEdges = (0 until size).map { row -> mk[row to 0, row to -1] }
            val rightEdges = (0 until size).map { row -> mk[row to size - 1, row to size] }
            val upperEdges = (0 until size).map { col -> mk[0 to col, -1 to col] }
            val lowerEdges = (0 until size).map { col -> mk[size - 1 to col, size to col] }

            // Step 3: Define the walls inside the maze
            val walls = mk[
                mk[1 to 0, 1 to 1], mk[2 to 0, 2 to 1], mk[3 to 0, 3 to 1],
                mk[1 to 1, 1 to 2], mk[2 to 1, 2 to 2], mk[3 to 1, 3 to 2],
                mk[3 to 1, 4 to 1], mk[0 to 2, 1 to 2], mk[1 to 2, 1 to 3],
                mk[2 to 2, 3 to 2], mk[2 to 3, 3 to 3], mk[2 to 4, 3 to 4],
                mk[4 to 2, 4 to 3], mk[1 to 3, 1 to 4], mk[2 to 3, 2 to 4]
            ]

            // Step 4: Combine edges and walls into obstacles
            val obstacles = upperEdges + lowerEdges + leftEdges + rightEdges + walls

            // Step 5: Remove obstacles from the maze connectivity
            for ((src, dst) in obstacles.map { it[0] to it[1] }) {
                maze[src]?.remove(dst)
                maze[dst]?.remove(src)
            }

            return maze
        }
    }
    private fun encode(rc: Pair<Int, Int>): Int = rc.first * size + rc.second

    private fun decode(s: Int): Pair<Int, Int> = (s / size) to (s % size)

    /**
     * Executes a single step in the environment by performing the given action.
     *
     * This method updates the current state of the environment based on the specified action,
     * computes the corresponding reward, and determines whether the episode has terminated.
     *
     * @param action The action to be executed within the environment.
     * @return A StepResult containing the new state after executing the action, the reward obtained,
     *         a flag indicating if the episode has terminated, a flag for truncation status (always false in this case),
     *         and an empty metadata map.
     */
    override fun step(action: Int): StepResult<Int> {
        val reward = computeReward(state)
        state = nextState(state, action)
        val terminated = state == goal
        return StepResult(encode(state), reward, terminated, false, emptyMap())
    }

    /**
     * Resets the environment to its initial state and optionally allows configuration
     * through a random seed or additional options.
     *
     * If `exploringStarts` is enabled, the initial state will be sampled until it does not
     * match the `goal` state. Otherwise, the state defaults to a predefined starting position.
     *
     * @param seed An optional integer seed for the random number generator, ensuring reproducibility
     *             of the initial state if provided.
     * @param options A map of optional configuration parameters for the environment reset process.
     *                These options can include additional customization for the reset behavior.
     * @return An `InitialState` object containing the reset initial state of the environment along
     *         with any associated metadata.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<Int> {
        state = start
        return InitialState(encode(state))
    }

    /**
     * Renders the current state of the maze environment as a visual representation.
     *
     * This method generates an image depicting the maze, including elements such as walls,
     * the agent's position, and the goal. The rendering process produces a byte array
     * representation of the image, suitable for visualization or further processing.
     *
     * @return A `Rendering` object, either:
     * - `Rendering.Empty` if rendering is disabled.
     * - `Rendering.RenderFrame` containing the rendered frame's dimensions and byte data.
     */
    override fun render(): Rendering {
        if (!render) return Rendering.Empty

        val cellSize = 80
        val imgSize = size * cellSize
        val image = BufferedImage(imgSize, imgSize, BufferedImage.TYPE_3BYTE_BGR)
        val g: Graphics2D = image.createGraphics()

        // Background
        g.color = Color(22, 36, 71)
        g.fillRect(0, 0, imgSize, imgSize)

        // Interior Walls – thin
        g.color = Color.WHITE
        g.stroke = BasicStroke(1f)
        for (row in 0 until size) {
            for (col in 0 until size) {
                val state = row to col
                val neighbors = maze[state].orEmpty().toSet()

                val x = col * cellSize
                val y = row * cellSize

                if (row - 1 to col !in neighbors) g.drawLine(x, y, x + cellSize, y)                       // Top
                if (row + 1 to col !in neighbors) g.drawLine(
                    x,
                    y + cellSize,
                    x + cellSize,
                    y + cellSize
                ) // Bottom
                if (row to col - 1 !in neighbors) g.drawLine(x, y, x, y + cellSize)                       // Left
                if (row to col + 1 !in neighbors) g.drawLine(x + cellSize, y, x + cellSize, y + cellSize) // Right
            }
        }

        // Border Walls – thick
        g.stroke = BasicStroke(3f)
        g.drawLine(0, 0, imgSize, 0)                      // Top
        g.drawLine(0, imgSize - 1, imgSize, imgSize - 1)  // Bottom
        g.drawLine(0, 0, 0, imgSize)                      // Left
        g.drawLine(imgSize - 1, 0, imgSize - 1, imgSize)  // Right

        // Goal
        g.color = Color(40, 199, 172)
        g.fillRect(goal.second * cellSize + 10, goal.first * cellSize + 10, cellSize - 20, cellSize - 20)

        // Agent
        g.color = Color(228, 63, 90)
        val cx = state.second * cellSize + cellSize / 2
        val cy = state.first * cellSize + cellSize / 2
        val r = (cellSize * 0.6 / 2).toInt()
        g.fillOval(cx - r, cy - r, r * 2, r * 2)

        g.dispose()

        // Convert to RGB byte array
        val pixels = IntArray(imgSize * imgSize)
        image.getRGB(0, 0, imgSize, imgSize, pixels, 0, imgSize)

        val rgbBytes = ByteArray(pixels.size * 3)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            rgbBytes[i * 3] = ((pixel shr 16) and 0xFF).toByte() // R
            rgbBytes[i * 3 + 1] = ((pixel shr 8) and 0xFF).toByte()  // G
            rgbBytes[i * 3 + 2] = (pixel and 0xFF).toByte()          // B
        }

        return Rendering.RenderFrame(width = imgSize, height = imgSize, bytes = rgbBytes.copyOf())
    }

    /**
     * Closes the maze environment and performs any necessary cleanup operations.
     *
     * This method is typically used to release resources associated with the
     * environment, ensuring proper termination and avoiding potential resource leaks.
     * After invoking this method, the environment should no longer be used.
     */
    override fun close() {

    }

    /**
     * Converts the current maze environment into a `TabularMDP` representation.
     *
     * This method constructs an MDP by defining the states, actions, reward function,
     * transition probabilities, and a discount factor for the maze environment.
     *
     * @param gamma The discount factor for the MDP, used to balance immediate and future rewards.
     *              It should be a value in the range [0, 1], where 0 focuses only on immediate rewards,
     *              and 1 considers future rewards fully.
     * @return A `TabularMDP` representation of the maze environment,
     *         containing states, actions, rewards, transitions, and the discount factor.
     */
    override fun asMDP(gamma: Double): TabularMDP {
        return TabularMDP(
            S = FiniteStates(size * size),
            A = FixedIntActions(4),
            RA = mk.d2arrayIndices<Double>(size * size, 4) { s, a ->
                val sRC  = decode(s)
                val nsRC = nextState(sRC, a)
                computeReward(nsRC)                       // reward for s'
            },
            TA = mk.d3arrayIndices<Double>(size * size, 4, size * size) { s, a, sP ->
                val state = decode(s)
                val nextState = nextState(state, a)
                if (encode(nextState) == sP) 1.0 else 0.0
            },
            gamma = gamma,
        )
    }

    /**
     * Determines the next state of the agent based on the current state and the specified action.
     *
     * The method computes the potential next state by applying the chosen action.
     * If the resulting state is valid (i.e., the action keeps the agent within the constraints
     * of the maze), it transitions to that state. Otherwise, the agent remains in the current state.
     *
     * @param state The current state represented as a 1D NDArray containing the row and column indices.
     * @param action The action to be performed, represented as an integer (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT).
     * @return The next state as a 1D NDArray of the same format as the input state. If the action
     *         results in an invalid transition, the current state is returned.
     */
    private fun nextState(state: Pair<Int, Int>, action: Int): Pair<Int, Int> {
        val (row, col) = state
        val nextState = when (action) {
            0 -> row - 1 to col // Move UP
            1 -> row to col + 1 // Move RIGHT
            2 -> row + 1 to col // Move DOWN
            3 -> row to col - 1 // Move LEFT
            else -> error("Action value not supported: $action")
        }

        return if (maze[row to col]?.contains(nextState) == true) {
            nextState
        } else {
            state
        }
    }

    /**
     * Computes the reward for the given state in the environment.
     *
     * The reward computation depends on whether shaped rewards are enabled.
     * If shaped rewards are enabled, the reward is based on the normalized
     * distance from the goal. Otherwise, a fixed reward is given for
     * reaching the goal, and a penalty is applied otherwise.
     *
     * @param state The current state represented as a 1D NDArray containing the row and column indices.
     * @return A Double representing the computed reward for the given state.
     */
    private fun computeReward(state: Pair<Int, Int>): Double {
        return if (state == goal) 0.0 else -1.0
    }
}