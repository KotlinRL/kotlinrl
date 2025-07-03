package io.github.kotlinrl.integration.display

import io.github.kotlinrl.core.env.Rendering.*
import java.awt.*
import java.awt.image.*
import javax.swing.*

class RenderDisplay(
    var image: BufferedImage
) : JPanel() {
    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        g.drawImage(image, 0, 0, this)
    }

    fun updateRenderFrame(renderFrame: RenderFrame) {
        val width = renderFrame.width
        val height = renderFrame.height
        val bgrBytes = renderFrame.bytes.toBGRArray(height, width)
        image.raster.setDataElements(0, 0, width, height, bgrBytes)
        repaint()
    }
}

fun createDisplay(renderFrame: RenderFrame, exitOnClose: Boolean = false): (RenderFrame) -> Unit {
    val width = renderFrame.width
    val height = renderFrame.height
    val bgrBytes = renderFrame.bytes.toBGRArray(height, width)
    val initialImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
    initialImage.raster.setDataElements(0, 0, width, height, bgrBytes)
    val panel = RenderDisplay(initialImage)
    panel.preferredSize = Dimension(renderFrame.width, renderFrame.height)
    val frame = JFrame("Env Rendering")
    frame.defaultCloseOperation = if(exitOnClose) JFrame.EXIT_ON_CLOSE else JFrame.DISPOSE_ON_CLOSE
    frame.add(panel)
    frame.pack()

    SwingUtilities.invokeLater {
        frame.setLocationRelativeTo(null)
        frame.isVisible = true
        frame.toFront()
        frame.requestFocus()
        frame.isAlwaysOnTop = true
    }
    return panel::updateRenderFrame
}

private fun ByteArray.toBGRArray(height: Int, width: Int): ByteArray {
    val rgbBytes = this
    val bgrBytes = ByteArray(rgbBytes.size)
    for (i in 0 until height * width) {
        bgrBytes[i*3]     = rgbBytes[i*3 + 2] // B
        bgrBytes[i*3 + 1] = rgbBytes[i*3 + 1] // G
        bgrBytes[i*3 + 2] = rgbBytes[i*3]     // R
    }
    return bgrBytes
}


