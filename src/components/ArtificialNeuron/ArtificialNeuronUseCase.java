package components.ArtificialNeuron;

import components.ArtificialNeuron.*;
import components.ArtificialNeuron.ArtificialNeuronKernal.ACTIVATION;

import java.util.*;

public class ArtificialNeuronUseCase {
    public static void main(String[] args){
        double[] inputs = {1, 5, 4, 7, 2, 7, 4};
        ArrayList<ArtificialNeuron> layer1 = new ArrayList<>();
        ArrayList<ArtificialNeuron> layer2 = new ArrayList<>();

        Random rand = new Random();

        layer1.add(new ArtificialNeuron1(inputs.length));
        layer1.add(new ArtificialNeuron1(inputs.length));
        layer1.add(new ArtificialNeuron1(inputs.length));

        for(ArtificialNeuron neuron : layer1){
            neuron.setAct(ACTIVATION.SILU);
        }

        layer2.add(new ArtificialNeuron1(inputs.length));
        layer2.add(new ArtificialNeuron1(inputs.length));
        layer2.add(new ArtificialNeuron1(inputs.length));

        for(ArtificialNeuron neuron : layer2){
            neuron.setAct(ACTIVATION.SIGMOID);
        }

        System.out.println("\n\nLayer 1: ");
        for(ArtificialNeuron n : layer1){
            System.out.println(n);
            System.out.println();
        }

        System.out.println("\n\nLayer 2: ");
        for(ArtificialNeuron n : layer2){
            System.out.println(n);
            System.out.println();
        }

        double[] layer1_outputs = new double[layer1.size()];
        int i = 0;
        for(ArtificialNeuron n : layer1){
            layer1_outputs[i] = n.forward(inputs);
            i++;
        }

        double[] layer2_outputs = new double[layer2.size()];
        i = 0;
        for(ArtificialNeuron n : layer2){
            layer2_outputs[i] = n.forward(layer1_outputs);
            i++;
        }

        double y = 0;
        for(double d : layer2_outputs){
            y += d;
        }

        System.out.println("Output of 2-Layer Nueral Network: " + y);
    }
}
