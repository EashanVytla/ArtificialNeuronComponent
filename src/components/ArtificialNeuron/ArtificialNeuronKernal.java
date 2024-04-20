package components.ArtificialNeuron;
public interface ArtificialNeuronKernal {
    enum ACTIVATION {
        SIGMOID,
        RELU,
        ELU,
        SILU
    }

    /**
     * returns number of input tensors
     *
     * @ensures weight is not modified
     * @return len(weights)
     */
    public int getNumInputs();

    /**
     * returns current value of weight at index
     *
     * @ensures weight is not modified
     * @requires 0 < index < input_size
     * @param int index: index of weight
     * @return weight[index]
     */
    public double getW(int index);

    /**
     * returns current value of bias
     *
     * @ensures bias is not modified
     * @return bias
     */
    public double getB();

    /**
     * sets the current value of weight at index
     *
     * @ensures this.weight[index] = w
     * @requires 0 < weight < 100
     * @param int index: index of weight
     *         double w: value to set as weight
     */
    public void setW(int index, double w);

    /**
     * sets the current value of bias at index
     *
     * @ensures this.bias = b
     * @requires 0 < b < 100
     * @param int index: index of weight
     *         double w: value to set as weight
     */
    public void setB(double b);

    /**
     * sets the current activation function
     *
     * @ensures this.act = act
     * @param ACTIVATION act: activation function
     */
    public void setAct(ACTIVATION act);
}
