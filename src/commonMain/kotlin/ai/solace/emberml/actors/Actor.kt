package ai.solace.emberml.actors

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.CoroutineScope

/**
 * Base interface for all messages in the actor system.
 */
interface ActorMessage

/**
 * Base interface for all actors in the system.
 *
 * @param T The type of messages this actor can receive.
 */
interface Actor<T : ActorMessage> {
    /**
     * Processes a message.
     */
    suspend fun receive(message: T)

    /**
     * Sends a message to this actor.
     */
    suspend fun send(message: T)

    /**
     * Stops the actor and cleans up resources.
     */
    suspend fun stop()
}

/**
 * Actor reference abstraction for external messaging.
 */
interface ActorRef<T : ActorMessage> {
    suspend fun tell(message: T)
    suspend fun <R : ActorMessage> ask(message: T, responseType: kotlin.reflect.KClass<R>): R
}

/**
 * Abstract base actor with built-in mailbox and message loop.
 */
abstract class BaseActor<T : ActorMessage> : Actor<T> {
    private val mailbox = Channel<T>(Channel.UNLIMITED)
    private var running = true

    override suspend fun send(message: T) {
        if (running) {
            mailbox.send(message)
        }
    }

    override suspend fun stop() {
        running = false
        mailbox.close()
    }

    suspend fun start() {
        while (running) {
            try {
                val message = mailbox.receive()
                receive(message)
            } catch (e: Exception) {
                onError(e)
            }
        }
    }

    protected open suspend fun onError(error: Exception) {
        println("Error in actor: ${error.message}")
    }
}