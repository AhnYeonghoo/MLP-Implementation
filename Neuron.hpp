#pragma once
#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>



class Neuron
{
private:
	std::vector<double> weight;
	double bias;
	mutable double lastV;
	double lastD;
	mutable std::vector<double> lastX;
	
public:
	Neuron(std::size_t input_size);
	~Neuron() {}
	
	// getter
	double getBias() const;
	std::vector<double> getWeight() const;
	double getLastV() const;
	double getLastD() const;
	const std::vector<double>& getLastX() const;

	// setter
	void setBias(double& bias);
	void setWeight(std::vector<double>& weight);

	// feed forward
	double compute(const std::vector<double>& x) const;

	// train
	// a�� learning rate
	void train(double a, const std::vector<std::pair<std::vector<double>, double>>
		& train_data);
	void train(double a, double e, const std::vector<double>& train_data);

	// bias�� weight �ʱ�ȭ �Լ�
	void initialized();

	// ����ġ�� ������ �����ϴ� �Լ�
	std::size_t inputSize() const;
};

