package components.ArtificialNeuron;
import java.util.ArrayList;
import java.util.Random;

public class ArtificialNeuronPrototype{
    private double w;
    private double b;
    private String act;

    public ArtificialNeuronPrototype(double w, double b){
        this.w = w;
        this.b = b;
        this.setAct("relu");;
    }

    public ArtificialNeuronPrototype(double w, double b, String act){
        this.w = w;
        this.b = b;
        this.setAct(act);
    }

    public ArtificialNeuronPrototype(){
        Random rand = new Random();
        this.w = rand.nextDouble();
        this.b = rand.nextDouble();
        this.setAct("relu");
    }

    public double getW(){
        return this.w;
    }

    public double getB(){
        return this.w;
    }

    public void setW(double w){
        this.w = w;
    }

    public void setB(double b){
        this.b = b;
    }

    public void setAct(String act){
        assert act == "relu" || act == "sigmoid";

        this.act = act;
    }

    public double forward(double[] X){
        double sum = 0.0;

        for(double x : X){
            sum += this.w * x + this.b;
        }

        if(this.act == "relu"){
            sum = this.relu(sum);
        }else if(this.act == "sigmoid"){
            sum = this.sigmoid(sum);
        }

        return sum;
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public String toString(){
        return "weight: " + this.w + "\nbias: " + this.b + "\nactivation:" + this.act;
    }
}

class Test {
	public static void main(String[] args){
        double[] inputs = {1, 5, 4, 7, 2, 7, 4};
        ArrayList<ArtificialNeuronPrototype> layer1 = new ArrayList<>();
        ArrayList<ArtificialNeuronPrototype> layer2 = new ArrayList<>();

        Random rand = new Random();

        layer1.add(new ArtificialNeuronPrototype());
        layer1.add(new ArtificialNeuronPrototype(rand.nextDouble(), rand.nextDouble(), "sigmoid"));
        layer1.add(new ArtificialNeuronPrototype(rand.nextDouble(), rand.nextDouble()));

        layer2.add(new ArtificialNeuronPrototype());
        layer2.add(new ArtificialNeuronPrototype(rand.nextDouble(), rand.nextDouble(), "sigmoid"));
        layer2.add(new ArtificialNeuronPrototype(rand.nextDouble(), rand.nextDouble()));

        System.out.println("\n\nLayer 1: ");
        for(ArtificialNeuronPrototype n : layer1){
            System.out.println(n);
            System.out.println();
        }

        System.out.println("\n\nLayer 2: ");
        for(ArtificialNeuronPrototype n : layer2){
            System.out.println(n);
            System.out.println();
        }

        double[] layer1_outputs = new double[layer1.size()];
        int i = 0;
        for(ArtificialNeuronPrototype n : layer1){
            layer1_outputs[i] = n.forward(inputs);
            i++;
        }

        double[] layer2_outputs = new double[layer2.size()];
        i = 0;
        for(ArtificialNeuronPrototype n : layer2){
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