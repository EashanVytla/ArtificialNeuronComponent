package components.ArtificialNeuron;

import static org.junit.Assert.assertEquals;
import java.util.Random;

import org.junit.Test;

import components.ArtificialNeuron.ArtificialNeuron1;
import components.ArtificialNeuron.ArtificialNeuronKernal.ACTIVATION;

public class ArtificialNeuronKernalTest {
    @Test
    public void testGetNumInputsNonZero(){
        int numInputs = 10;
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs);

        assertEquals(neuron.getNumInputs(), numInputs);
    }

    @Test
    public void testGetNumInputsZero(){
        int numInputs = 0;
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs);

        assertEquals(neuron.getNumInputs(), numInputs);
    }

    @Test
    public void testGetWAll(){
        int numInputs = 10;
        Random rand = new Random(100);
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        for(int i = 0; i < numInputs; i++){
            assertEquals(neuron.getW(i), rand.nextDouble(), 0.001);
        }
    }

    @Test
    public void testSetWAll(){
        int numInputs = 10;
        double[] expectedVals = {1.0, 0.5, 0.1, 0.4, 0.2, 0.3, 0.6, 0.7, 0.8, 0.35};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs);

        for(int i = 0; i < numInputs; i++){
            neuron.setW(i, expectedVals[i]);
        }

        for(int i = 0; i < numInputs; i++){
            assertEquals(neuron.getW(i), expectedVals[i], 0.001);
        }
    }

    @Test
    public void testGetB(){
        int numInputs = 10;
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs);

        assertEquals(neuron.getB(), 0.0, 0.001);
    }

    @Test
    public void testForward100(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        double result = neuron.forward(X);
        assertEquals(result, 0.9524979744396342, 0.001);
    }

    @Test
    public void testForward97(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 97);

        double result = neuron.forward(X);
        assertEquals(result, 0.957213515346698, 0.001);
    }

    @Test
    public void testForward56(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 56);

        double result = neuron.forward(X);
        assertEquals(result, 0.9568929905569626, 0.001);
    }

    @Test
    public void testForward20(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 20);

        double result = neuron.forward(X);
        assertEquals(result, 0.9189442479765013, 0.001);
    }

    @Test
    public void testForward100Silu(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        neuron.setAct(ACTIVATION.SILU);

        double result = neuron.forward(X);
        assertEquals(result, 2.855889561370592, 0.001);
    }

    @Test
    public void testForward97Silu(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 97);

        neuron.setAct(ACTIVATION.SILU);

        double result = neuron.forward(X);
        assertEquals(result, 2.966373965795305, 0.001);
    }

    @Test
    public void testForward56Silu(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 56);

        neuron.setAct(ACTIVATION.SILU);

        double result = neuron.forward(X);
        assertEquals(result, 2.9748321852176076, 0.001);
    }

    @Test
    public void testForward20Silu(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 20);

        neuron.setAct(ACTIVATION.SILU);

        double result = neuron.forward(X);
        assertEquals(result, 2.2312777217799846, 0.001);
    }

    @Test
    public void testForward100RELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        neuron.setAct(ACTIVATION.RELU);

        double result = neuron.forward(X);
        assertEquals(result, 2.9983156269185196, 0.001);
    }

    @Test
    public void testForward97RELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 97);

        neuron.setAct(ACTIVATION.RELU);

        double result = neuron.forward(X);
        assertEquals(result, 3.1000059516254974, 0.001);
    }

    @Test
    public void testForward56RELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 56);

        neuron.setAct(ACTIVATION.RELU);

        double result = neuron.forward(X);
        assertEquals(result, 3.107804202012482, 0.001);
    }

    @Test
    public void testForward20RELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 20);

        neuron.setAct(ACTIVATION.RELU);

        double result = neuron.forward(X);
        assertEquals(result, 2.4280882400572374, 0.001);
    }

    @Test
    public void testForward100ELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 100);

        neuron.setAct(ACTIVATION.ELU);

        double result = neuron.forward(X);
        assertEquals(result, 2.9983156269185196, 0.001);
    }

    @Test
    public void testForward97ELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 97);

        neuron.setAct(ACTIVATION.ELU);

        double result = neuron.forward(X);
        assertEquals(result, 3.1000059516254974, 0.001);
    }

    @Test
    public void testForward56ELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 56);

        neuron.setAct(ACTIVATION.ELU);

        double result = neuron.forward(X);
        assertEquals(result, 3.107804202012482, 0.001);
    }

    @Test
    public void testForward20ELU(){
        int numInputs = 10;
        double[] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        ArtificialNeuron neuron = new ArtificialNeuron1(numInputs, 20);

        neuron.setAct(ACTIVATION.ELU);

        double result = neuron.forward(X);
        assertEquals(result, 2.4280882400572374, 0.001);
    }
}
