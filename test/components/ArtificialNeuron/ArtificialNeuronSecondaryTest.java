package components.ArtificialNeuron;

import static org.junit.Assert.assertEquals;
import java.util.Random;

import org.junit.Test;

import components.ArtificialNeuron.ArtificialNeuronSecondary;


public class ArtificialNeuronSecondaryTest {
    @Test
    public void testEquals(){
        int numInputs = 10;
        ArtificialNeuron neuron1 = new ArtificialNeuron1(numInputs, 100);
        ArtificialNeuron neuron2 = new ArtificialNeuron1(numInputs, 100);

        assertEquals(neuron1, neuron2);
    }

    @Test
    public void testToString(){
        int numInputs = 10;
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        assertEquals(neuron.toString(), "Neuron\nWeights:\n0.7220096548596434\n0.19497605734770518\n0.6671595726539502\n0.7784408674101491\n0.6186076060240648\n0.62309699131219\n0.23675163488485773\n0.48722715836911057\n0.6804221244148292\n0.524545450315388\nBias: 0.0");
    }

    @Test
    public void testForward(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        assertEquals(neuron.forward(X), 0.9524979744396342, 0.001);
    }
}
