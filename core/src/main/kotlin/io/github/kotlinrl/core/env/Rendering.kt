package io.github.kotlinrl.core.env

sealed class Rendering {
    data class Text(val ansi: String) : Rendering()
    object Empty : Rendering()
    data class RenderFrame(
        val width: Int,
        val height: Int,
        val bytes: ByteArray
    ) : Rendering() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as RenderFrame

            if (width != other.width) return false
            if (height != other.height) return false
            if (!bytes.contentEquals(other.bytes)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = width
            result = 31 * result + height
            result = 31 * result + bytes.contentHashCode()
            return result
        }
    }
}