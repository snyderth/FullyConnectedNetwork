package FullyConnectedNetwork;
import java.util.ArrayList;


public class Network{

	private final int[] NETWORK_LAYER_SIZES;
	private final int		INPUT_SIZE;
	private final int 	OUTPUT_SIZE;
	private final int		NETWORK_SIZE;
	private Perceptron[][] LAYERS;
	private double[] avg_cost;
	private double iteration;
	private boolean TRAINING;
	/**
	*	Constructor for the full network
	* @param NETWORK_LAYER_SIZES Takes an array whose length is the number of
	*	layers and whose indeces contain the number of perceptrons in each layer
	* @param train True if training the network, false if network is performing
	*/
	public Network(int[] NETWORK_LAYER_SIZES, boolean train)
	{
		this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
		this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
		this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_LAYER_SIZES.length - 1];
		this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
		this.LAYERS = new Perceptron[NETWORK_SIZE][];
		for(int i = 0; i < NETWORK_SIZE; i++)
			this.LAYERS[i] = new Perceptron[NETWORK_LAYER_SIZES[i]];
		//Instantiate new perceptrons in each layer
		for(int i = 0; i < NETWORK_SIZE; i++)
		{
			for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++)
			{
				if(i == 0)
				{//First layer
					LAYERS[i][j] = new Perceptron();
				}
				else
				{
					LAYERS[i][j] = new Perceptron(LAYERS[i-1]);
				}
			}
		}
		this.avg_cost = new double[OUTPUT_SIZE];
		this.iteration = 0.0;
		TRAINING = train;
	}


	/**
	*	A function that associates a cost with the guesses of the network. For training use.
	*	@param outputs An array of double values that are the result of the computation.
	* @param expected A double value of the expected results
	*/
	public void compute_cost(double[] outputs, double[] expected)
	{
		iteration += 1.0;
		double cost = 0;
		for(int i = 0; i < outputs.length; i++)
		{
			//This sum is small when the network classifies confidently and correctly
			//This sum is large when the network classifies incorrectly or not confidently
			cost = Math.pow((outputs[i]-expected[i]), 2);

			avg_cost[i] = ( ( (iteration - 1.0) / iteration) * cost) + ( (1.0 / iteration) * cost);
		}
	}


	/**
	*	A function that computes on the given data and feeds the outputs into a cost function
	* @param input A double value given for the first layer of the network
	*	@param expected An array of double values. It should have the same length as the output
	*	layer and the expected index should have a value of 1.00 while all others should have
	*	a value of 0.00
	*/
	public void compute(double[] input, double[] expected)
	{
		double[] outputs = new double[OUTPUT_SIZE];
		for(int i = 0; i < NETWORK_SIZE; i++)
		{
			for(int j = 0; j < NETWORK_LAYER_SIZES[i]; j++)
			{
				LAYERS[i][j].compute(input[i]);
			}
		}
		for(int i = 0; i < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; i++)
		{
			outputs[i] = LAYERS[NETWORK_SIZE-1][i].get_output();
		}
		if(TRAINING)
		{
			train_network(outputs, expected);
		}

	}


	/**
	*	A function to train the network using supervised learning
	* @param outputs The outputs of the final layer of perceptrons
	* @param expected The outputs that we want the network to give
	*/
	public void train_network(double[] outputs, double[] expected)
	{

		for(int i = 0; i < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; i++)
		{//This only deals with the final layer, what about the layers before?
			LAYERS[NETWORK_SIZE-1][i].train(expected[i]);
		}
	}


	/**
	*	A function that prints the result of the network.
	*/
	public void print_output()
	{
		for(int i = 0; i < OUTPUT_SIZE; i++)
		{
			System.out.println( "Output " + i + ": " + LAYERS[NETWORK_SIZE-1][i].get_output());
		}
	}




	/*Checking command line args*/
	private static boolean good_args(String[] args)
	{
		System.out.println("Checking command line arguments...");
		for(int i = 0; i < args.length; i++){
			for(int j = 0; j <= args[i].length() - 1; j++){
				if(args[i].charAt(j) > '9' || args[i].charAt(j) < '0')
				{
					System.out.println("Invalid character in command line argument \"" + args[i] + "\"");
					return false;
				}
			}
		}
		System.out.println("Done.");
		return true;
	}


	public double get_avg_cost(){
		double cost = 0;
		for(int i = 0; i < avg_cost.length; i++)
		{
			cost += avg_cost[i];
		}
		cost/= avg_cost.length;
		return cost;
	}


	/*main method*/
	public static void main(String[] args)
	{
		if(args.length > 0)
		{
			if(!good_args(args))
				return;
			else
			{
				int[] arr = new int[args.length];
				for(int i = 0; i < args.length; i++)
					arr[i] = Integer.parseInt(args[i]);
				Network net = new Network(arr, true);
				double[] inputs = new double[arr[0]];
				for(int i = 0; i < arr[0]; i++)
				{
					inputs[i] = Math.random();
				}
				double[] expected = new double[arr[arr.length - 1]];
				for(int i = 0; i < expected.length; i++)
					expected[i] = 0;
				expected[expected.length/2] = 1.0;
				net.compute(inputs, expected);
				net.print_output();
				System.out.println("Error: " + net.get_avg_cost());
			}

		}else{
			System.out.println("Please provide integer arguments for the number of perceptrons in each layer of the network.");
		}
	}

}
