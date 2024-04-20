package components.ArtificialNeuron;

import java.util.Random;

public class ArtificialNeuron1 extends ArtificialNeuronSecondary {
    private double[] weights;
    private double bias;
    private ACTIVATION activation;

    public ArtificialNeuron1(int inputSize, int randSeed) {
        this.weights = new double[inputSize];
        this.bias = 0.0;
        this.activation = ACTIVATION.SIGMOID;

        Random random = new Random(randSeed);

        for(int i = 0; i < inputSize; i++){
            this.weights[i] = random.nextDouble();
        }
    }

    public ArtificialNeuron1(int inputSize) {
        this.weights = new double[inputSize];
        this.bias = 0.0;
        this.activation = ACTIVATION.SIGMOID;

        Random random = new Random();

        for(int i = 0; i < inputSize; i++){
            this.weights[i] = random.nextDouble();
        }
    }

    @Override
    public int getNumInputs() {
        return this.weights.length;
    }

    @Override
    public double getW(int index) {
        return this.weights[index];
    }

    @Override
    public double getB() {
        return this.bias;
    }

    @Override
    public void setW(int index, double w) {
        this.weights[index] = w;
    }

    @Override
    public void setB(double b) {
        this.bias = b;
    }

    @Override
    public void setAct(ACTIVATION act) {
        this.activation = act;
    }

    @Override
    public double forward(double[] X) {
        double output = 0.0;
        for (int i = 0; i < X.length; i++) {
            output += this.weights[i] * X[i];
        }
        output += this.bias;

        switch (this.activation) {
            case SIGMOID:
                return this.sigmoid(output);
            case RELU:
                return this.relu(output);
            case ELU:
                return this.elu(output);
            case SILU:
                return this.silu(output);
            default:
                return output;
        }
    }

    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double relu(double x) {
        return Math.max(0.0, x);
    }

    public double elu(double x) {
        return (x >= 0) ? x : Math.exp(x) - 1;
    }

    public double silu(double x) {
        return x / (1.0 + Math.exp(-x));
    }
}
