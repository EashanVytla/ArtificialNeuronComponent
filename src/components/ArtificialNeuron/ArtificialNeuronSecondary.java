package components.ArtificialNeuron;
public abstract class ArtificialNeuronSecondary implements ArtificialNeuron{
    @Override
    public double forward(double[] X){
        double output = 0.0;

        for (int i = 0; i < X.length; i++) {
            output += this.getW(i) * X[i];
        }

        output += this.getB();

        return output;
    }

    @Override
    public String toString(){
        String result = "Neuron\nWeights:\n";
        for (int i = 0; i < this.getNumInputs(); i++) {
            result += this.getW(i) + "\n";
        }
        result += "Bias: " + this.getB();
        return result;
    }

    @Override
    public boolean equals(Object other){
        boolean isEqual = true;
        for (int i = 0; i < this.getNumInputs(); i++) {
            if (this.getW(i) != ((ArtificialNeuronKernal)(other)).getW(i)) {
                isEqual = false;
                break;
            }
        }
        isEqual = isEqual && this.getB() == ((ArtificialNeuronKernal)(other)).getB();
        return isEqual;
    }
}
