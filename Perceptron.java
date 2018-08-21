package FullyConnectedNetwork;


public class Perceptron{
	private boolean first_layer;
	private Perceptron[] inputs;
	private double[] weights;
	private double bias;
	private double OUTPUT;
	private double[] delta_weight;
	private int INPUT_SIZE;

	/**
	*	A constructor for all perceptrons except first layer
	*	@param inputs The array of the perceptrons in the previous layer
	*/
	public Perceptron(Perceptron[] inputs)
	{

			first_layer = false;
			this.inputs = inputs;
			INPUT_SIZE = this.inputs.length;
			this.weights = new double[INPUT_SIZE];
			this.delta_weight = new double[INPUT_SIZE];
			for(int i = 0; i < this.weights.length; i++)
			{
					this.weights[i] = Math.random();
			}
			bias = Math.random();

	}

	/**
	*	A constructor for the first layer of the network
	*/
	public Perceptron()
	{
			first_layer = true;
	}



	/**
	*	A function to comput the output of all perceptrons not in the
	*	first layer. Does not take any inputs, will call the previous
	* compute functions to sum all the outputs in the previous layer
	* @param input A double value that is the data to be calculated on
	*/
	public void compute(double input)
	{
			double out = 0;
			//First layer
			if(first_layer){
				out = input;
			}else{ //All other layers
				for(int i = 0; i < inputs.length; i++)
				{
					out += (inputs[i].get_output() * weights[i]);
				}
				out += bias;
			}
			//Activation function
			out = sigmoid(out);
			OUTPUT = out;
	}


	/**
	*	A function that implements the sigmoid function: 1/(1+exp(-x))
	*	@param input The summed and weighted inputs of the Perceptron
	* @return A double value that is the output of the perceptron
	*/
	private double sigmoid(double input)
	{
		double out = 1 / (1 + Math.exp(-input));
		return out;
	}


	/**
	*	A function that implements sigmoid primve.
	*	<dl>
	*	<dt><b>See Also:</b>
	*	<dd><a href="https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x">Stack Exchange</a>
	*	</dl>
	* @param input A double value that represents the activation of the previous
	*	perceptron multiplied by the weight of the current perceptron and added summed
	* with the bias
	*	@return A double that represents the derivative of the sigmoid function
	*/
	public double sigmoid_prime(double input)
	{
		double out = sigmoid(input)*(1-sigmoid(input));
		return out;
	}


	/**
	*	A function that recursively trains all the weights that went into this perceptron, called by {@link FullyConnectedNetwork.Perceptron#train train}
	* @param dCdivA A double value that is the dC/dw of the previous perceptron divided by the activation of this perceptron
	* @param prev_weight A double value that is the weight of link between this perceptron and the previous percetron (one farther down the line)
	*/
	public void train_weights(double dCdivA, double prev_weight)
	{
		if(!first_layer)
		{
			for(int i = 0; i < INPUT_SIZE; i++)
			{
				//calculate dC/dw^(x)
				double dC = inputs[i].get_output() *
								sigmoid_prime(weights[i]*inputs[i].get_output() + this.bias) *
								prev_weight *
								dCdivA;
				delta_weight[i] = dC;

				//take off the activation of the input for feeding to the next perceptron
				//training function
				double dCdivd = dC / inputs[i].get_output();
				this.inputs[i].train_weights(dCdivd, weights[i]);
			}
			for(int i = 0; i < weights.length; i++)
			{
				this.weights[i] += delta_weight[i];
			}
		}


	}


	/**
	*	A function that recursively trains all the weights that went into this perceptron
	*	@param expected A double value holding the expected output of this perceptron
	*/
	public void train(double expected)
	{

		for(int i = 0; i < INPUT_SIZE; i++)
		{
			//calculate dC/dw^(L)
			double dC = inputs[i].get_output() *
							sigmoid_prime(weights[i]*inputs[i].get_output() + this.bias) *
							2 * (expected - OUTPUT);
			//Store the delta weight for later when we add to weights
			delta_weight[i] = dC;
			//divide dC/dw^(L) by activation of the input
			double dCdivA = dC / inputs[i].get_output();
			//call train function recursively passing the sensitivity of the weight
			//divided by the input activation
			this.inputs[i].train_weights(dCdivA, weights[i]);

		}
	}


	//setters
	public void add_input(Perceptron p)
	{
		Perceptron[] newInputs = new Perceptron[this.inputs.length + 1];
		for(int i = 0; i < this.inputs.length; i++)
			newInputs[i] = inputs[i];
		newInputs[newInputs.length - 1] = p;
		this.inputs = newInputs;
	}
	public void remove_input(int index){
		Perceptron[] newInputs = new Perceptron[this.inputs.length - 1];
		for(int i = 0; i < this.inputs.length; i++)
		{
			if(i < index)
				newInputs[i] = this.inputs[i];
			else if (i > index)
				newInputs[i-1] = this.inputs[i];
		}
		this.inputs = newInputs;
	}
	public void set_bias(double bias){this.bias = bias;}
	public void set_weights(double[] weights){this.weights = weights;}

	//getters
	public double[] get_weights(){return this.weights;}
	public Perceptron[] get_inputs(){return this.inputs;}
	public boolean is_first_layer(){return this.first_layer;}
	public double get_bias(){return this.bias;}
	public double get_output(){return this.OUTPUT;}
}
