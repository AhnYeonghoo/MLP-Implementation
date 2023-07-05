#pragma once
#include "Neuron.hpp"

class Network
{
private:
	std::vector<std::vector<Neuron>> layers;

public:
	Network(const std::vector<std::size_t>& layers);
	
	std::vector<double> advancedCompute(const std::vector<double>& x);
	
	void advancedTrain(double a, const std::vector<std::pair<std::vector<double>, std::vector<double>>>
		& train_data);
};