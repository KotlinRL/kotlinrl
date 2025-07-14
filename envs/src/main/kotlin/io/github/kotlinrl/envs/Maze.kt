package io.github.kotlinrl.envs

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.space.*
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Graphics2D
import java.awt.image.*
import kotlin.random.*

class Maze(
    val exploringStarts: Boolean = false,
    val shapedRewards: Boolean = false,
    val render: Boolean = true,
    override val size: Int = 5,
    override val metadata: Map<String, Any> = emptyMap(),
    seed: Int? = null,
) : ModelBasedEnv {
    enum class Action(val value: Int) {
        UP(0),
        RIGHT(1),
        DOWN(2),
        LEFT(3);
    }

    override val observationSpace: MultiDiscrete = MultiDiscrete(nvec = intArrayOf(size, size), seed = seed)
    override val actionSpace: Discrete = Discrete(n = 4, start = 0, seed = seed)
    override val random: Random  = seed?.let { Random(it) } ?: Random.Default
    private var state = intArrayOf(size - 1, size - 1)
    override val goal = intArrayOf(size - 1, size - 1)
    private val maze = createMaze(size)
    private val distances = computeDistances(size, goal.toList(), maze)

    private companion object {
        fun setValueAt(
            matrix: List<List<Double>>,
            row: Int,
            col: Int,
            value: Double
        ): List<List<Double>> {
            return matrix.mapIndexed { r, rowList ->
                if (r == row) {
                    rowList.mapIndexed { c, oldValue ->
                        if (c == col) value else oldValue
                    }
                } else {
                    rowList
                }
            }
        }


        fun createMaze(size: Int): Map<List<Int>, List<List<Int>>> {
            val maze = (0 until size).flatMap { row ->
                (0 until size).map { col ->
                    listOf(row, col) to mutableListOf(
                        listOf(row - 1, col), // Above
                        listOf(row + 1, col), // Below
                        listOf(row, col - 1), // Left
                        listOf(row, col + 1)  // Right
                    ).filter { (r, c) ->
                        // Filter out coordinates that are outside the bounds of the grid
                        r in 0 until size && c in 0 until size
                    }.toMutableList()
                }
            }.toMap().toMutableMap()

            // Step 2: Define the edges of the maze
            val leftEdges = (0 until size).map { row -> listOf(listOf(row, 0), listOf(row, -1)) }
            val rightEdges = (0 until size).map { row -> listOf(listOf(row, size - 1), listOf(row, size)) }
            val upperEdges = (0 until size).map { col -> listOf(listOf(0, col), listOf(-1, col)) }
            val lowerEdges = (0 until size).map { col -> listOf(listOf(size - 1, col), listOf(size, col)) }

            // Step 3: Define the walls inside the maze
            val walls = listOf(
                listOf(listOf(1, 0), listOf(1, 1)), listOf(listOf(2, 0), listOf(2, 1)), listOf(listOf(3, 0), listOf(3, 1)),
                listOf(listOf(1, 1), listOf(1, 2)), listOf(listOf(2, 1), listOf(2, 2)), listOf(listOf(3, 1), listOf(3, 2)),
                listOf(listOf(3, 1), listOf(4, 1)), listOf(listOf(0, 2), listOf(1, 2)), listOf(listOf(1, 2), listOf(1, 3)),
                listOf(listOf(2, 2), listOf(3, 2)), listOf(listOf(2, 3), listOf(3, 3)), listOf(listOf(2, 4), listOf(3, 4)),
                listOf(listOf(4, 2), listOf(4, 3)), listOf(listOf(1, 3), listOf(1, 4)), listOf(listOf(2, 3), listOf(2, 4)),
            )

            // Step 4: Combine edges and walls into obstacles
            val obstacles = upperEdges + lowerEdges + leftEdges + rightEdges + walls

            // Step 5: Remove obstacles from the maze connectivity
            for ((src, dst) in obstacles.map { it[0] to it[1] }) {
                maze[src]?.remove(dst)
                maze[dst]?.remove(src)
            }

            return maze
        }

        fun computeDistances(
            size: Int,
            goal: List<Int>,
            maze: Map<List<Int>, List<List<Int>>>): List<List<Double>> {
            var distances = List(size) { List(size) { Double.POSITIVE_INFINITY } }
            val visited = mutableSetOf<List<Int>>()
            val (startRow, startCol) = goal
            distances = setValueAt(distances, startRow, startCol, 0.0)

            while (visited != maze.keys) {
                val closest = maze.keys.filter { it !in visited }
                    .minByOrNull {
                        val (r, c) = it
                        distances[r][c]
                    }
                    ?: break

                visited.add(closest)

                for (neighbor in maze[closest] ?: emptyList()) {
                    val (row, col) = neighbor
                    val newDistance = distances[row][col] + 1
                    distances = setValueAt(distances, row, col, newDistance)
                }
            }
            return distances

        }
    }

    override fun step(action: Int): Transition<IntArray> {
        val reward = computeReward(state, action)
        state = nextState(state, action)
        val terminated = state.contentEquals(goal)
        return Transition(state, reward, terminated, false, emptyMap())
    }

    override fun reset(seed: Int?, options: Map<String, String>?): InitialState<IntArray> {
        if(exploringStarts) {
            while(state[0] == goal[0] && state[1] == goal[1]) {
                state = observationSpace.sample()
            }
        } else {
            state = intArrayOf(0, 0)
        }
        return InitialState(state)
    }

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
                val state = listOf(row, col)
                val neighbors = maze[state].orEmpty().toSet()

                val x = col * cellSize
                val y = row * cellSize

                if (listOf(row - 1, col) !in neighbors) g.drawLine(x, y, x + cellSize, y)                       // Top
                if (listOf(row + 1, col) !in neighbors) g.drawLine(x, y + cellSize, x + cellSize, y + cellSize) // Bottom
                if (listOf(row, col - 1) !in neighbors) g.drawLine(x, y, x, y + cellSize)                       // Left
                if (listOf(row, col + 1) !in neighbors) g.drawLine(x + cellSize, y, x + cellSize, y + cellSize) // Right
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
        g.fillRect(goal[1] * cellSize + 10, goal[0] * cellSize + 10, cellSize - 20, cellSize - 20)

        // Agent
        g.color = Color(228, 63, 90)
        val cx = state[1] * cellSize + cellSize / 2
        val cy = state[0] * cellSize + cellSize / 2
        val r = (cellSize * 0.6 / 2).toInt()
        g.fillOval(cx - r, cy - r, r * 2, r * 2)

        g.dispose()

        // Convert to RGB byte array
        val pixels = IntArray(imgSize * imgSize)
        image.getRGB(0, 0, imgSize, imgSize, pixels, 0, imgSize)

        val rgbBytes = ByteArray(pixels.size * 3)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            rgbBytes[i * 3]     = ((pixel shr 16) and 0xFF).toByte() // R
            rgbBytes[i * 3 + 1] = ((pixel shr 8) and 0xFF).toByte()  // G
            rgbBytes[i * 3 + 2] = (pixel and 0xFF).toByte()          // B
        }

        return Rendering.RenderFrame(width = imgSize, height = imgSize, bytes = rgbBytes)
    }

    override fun close() {

    }

    override fun nextState(state: IntArray, action: Int): IntArray {
        val (row, col) = state
        val nextState = when (action) {
            0 -> intArrayOf(row - 1, col) // Move UP
            1 -> intArrayOf(row, col + 1) // Move RIGHT
            2 -> intArrayOf(row + 1, col) // Move DOWN
            3 -> intArrayOf(row, col - 1) // Move LEFT
            else -> error("Action value not supported: $action")
        }

        return if (maze[listOf(row, col)]?.contains(nextState.toList()) == true) {
            nextState
        } else {
            state
        }
    }

    override fun computeReward(state: IntArray, action: Int): Double {
        val nextState = nextState(state, action)
        return if(shapedRewards) {
            val goalDistance = distances[nextState[0]][nextState[1]]
            val maxDistance = distances.flatten().maxOrNull() ?: Double.POSITIVE_INFINITY
            - (goalDistance / maxDistance)
        } else {
            if (state.contentEquals(goal)) 0.0 else -1.0
        }
    }

    override fun simulateStep(action: Int): Transition<IntArray> {
        val reward = computeReward(state, action)
        val nextState = nextState(state, action)
        val terminated = state.toList() == goal
        return Transition(nextState, reward, terminated, false, emptyMap())
    }

    override fun actionProbabilities(state: IntArray): DoubleArray = DoubleArray(4) { 1.0 / size }

    override fun stateActionList(state: IntArray): List<Int> = Action.entries.map { it.value }
}