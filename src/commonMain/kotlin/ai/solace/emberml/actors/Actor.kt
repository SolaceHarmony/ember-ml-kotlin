package ai.solace.emberml.actors

import kotlinx.coroutines.channels.Channel

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
     *
     * @param message The message to process.
     */
    suspend fun receive(message: T)
    
    /**
     * Sends a message to this actor.
     *
     * @param message The message to send.
     */
    suspend fun send(message: T)
    
    /**
     * Stops the actor.
     */
    suspend fun stop()
}

/**
 * Actor reference that can be used to send messages to an actor.
 *
 * @param T The type of messages the referenced actor can receive.
 */
interface ActorRef<T : ActorMessage> {
    /**
     * Sends a message to the referenced actor.
     *
     * @param message The message to send.
     */
    suspend fun tell(message: T)
    
    /**
     * Sends a message and waits for a response.
     *
     * @param message The message to send.
     * @param responseType The expected response type.
     * @return The response message.
     */
    suspend fun <R : ActorMessage> ask(message: T, responseType: kotlin.reflect.KClass<R>): R
}

/**
 * Basic actor implementation.
 *
 * @param T The type of messages this actor can receive.
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
    
    /**
     * Starts the actor's message processing loop.
     */
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
    
    /**
     * Called when an error occurs during message processing.
     *
     * @param error The error that occurred.
     */
    protected open suspend fun onError(error: Exception) {
        // Default implementation: log the error
        println("Error in actor: ${error.message}")
    }
}