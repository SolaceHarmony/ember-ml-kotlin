# Neuron Model vs. Neuron Role in Wired RNNs

This diagram illustrates that in a wired RNN layer (like LTC or a wired CfC using a NeuronMap), all internal units use the *same* underlying mathematical neuron model (e.g., LTCCell dynamics). The *role* of each unit (Sensory Input, Interneuron, Command Neuron, Motor Neuron/Output) is determined by the connections defined in the `NeuronMap`. Feedback occurs through the recurrent connections specified in the map and the cell's inherent state update mechanism.

```mermaid
graph LR
    subgraph InputOutput
        Input[Input Tensor features]
        Output[Output Tensor features]
    end

    subgraph WiredRNLayer [Wired RNN Layer e.g., LTC]
        Map[NeuronMap Defines Structure]
        Cell[Neuron Model e.g., LTCCell Applies Dynamics to ALL Units]

        subgraph InternalUnits [Internal Units state_size]
            Inter[Interneurons]
            Command[Command Neurons]
            Motor[Motor Neurons output_size subset]
        end

        Map -- defines --> InternalUnits
        Map -- defines --> SensoryConnections[Sensory Connections input_size to units]
        Map -- defines --> RecurrentConnections[Recurrent Connections units to units]
        Map -- defines --> OutputMapping[Output Mapping units to output_size]

        Input -- SensoryConnections --> Inter
        Input -- SensoryConnections --> Command

        Inter -- RecurrentConnections --> Inter
        Inter -- RecurrentConnections --> Command
        Inter -- RecurrentConnections --> Motor
        Command -- RecurrentConnections --> Command
        Command -- RecurrentConnections --> Motor

        Cell -- applies dynamics to --> Inter
        Cell -- applies dynamics to --> Command
        Cell -- applies dynamics to --> Motor

        Motor -- OutputMapping --> Output
    end

    classDef box fill:#eee,stroke:#333,stroke-width:1px;
    classDef concept fill:#lightblue,stroke:#333,stroke-width:1px;
    classDef neuron fill:#lightyellow,stroke:#333,stroke-width:1px;
    classDef io fill:#lightgreen,stroke:#333,stroke-width:1px;

    class WiredRNLayer,InternalUnits,InputOutput box;
    class Map,Cell concept;
    class Inter,Command,Motor neuron;
    class Input,Output io;