package components.ArtificialNeuron;
public interface ArtificialNeuron extends ArtificialNeuronKernal{
    /**
     * runs a forward pass with given X inputs
     *
     * @ensures this.act = act
     * @requires size(X) = input size
     * @param double[] X: array of inputs
     * @return double output of the nueron
     */
    public double forward(double[] X);
}
