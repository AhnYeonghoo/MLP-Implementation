#pragma once
#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>

class Neuron
{
private:
	std::vector<double> weight;
	double bias;
	
public:
	Neuron(std::size_t input_size);
	~Neuron() {}
	
	// getter
	double getBias() const;
	std::vector<double> getWeight() const;

	// setter
	void setBias(double& bias);
	void setWeight(std::vector<double>& weight);

	// activation function
	int stepFunction(int s) const;

	// compute
	double compute(const std::vector<double>& x) const;

	// train
	// a´Â learning rate
	void train(double a, const std::vector<std::pair<std::vector<double>, double>>
		& train_data);

	void initialized();
};

