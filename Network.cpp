#include "Network.hpp"

Network::Network(const std::vector<std::size_t>& layers)
{
	for (std::size_t i = 1; i < layers.size(); i++)
	{
		std::vector<Neuron> layer;
		for (std::size_t j = 0; j < layers[i]; j++)
		{
			layer.push_back(Neuron(layers[i - 1]));
		}
		this->layers.push_back(layer);
	}
}

std::vector<double> Network::advancedCompute(const std::vector<double>& x)
{
    if (x.size() != this->layers.at(0)[0].inputSize())
    {
        std::cerr << "입력층 사이즈가 옳지 않습니다" << std::endl;
    }

    std::vector<double> result;
    std::vector<double> next_layer = x;

    for (std::size_t i = 0; i < this->layers.size(); i++)
    {
        result.clear();
        for (std::size_t j = 0; j < this->layers[i].size(); j++)
        {
            result.push_back(this->layers[i][j].compute(next_layer));
        }
        next_layer = result;
    }

    return result;
}

void Network::advancedTrain(double a, const std::vector<std::pair<std::vector<double>, std::vector<double>>>
    & train_data)
{
    for (std::size_t i = 0; i < train_data.size(); i++)
    {
        // 출력 레이어 학습
        std::vector<double> o = advancedCompute(train_data[i].first);
        std::vector<double> e;
        
        if (o.size() != train_data[i].second.size())
        {
            std::cerr << "o.size() != train_data[i].second.size()";
        }
        
        for (std::size_t j = 0; j < o.size(); j++)
        {
            e.push_back(train_data[i].second[j] - o[j]);
        }

        std::vector<double> d;

        for (std::size_t j = 0; j < this->layers[this->layers.size() - 1].size(); ++j)
        {
            this->layers[this->layers.size() - 1][j].train(a, e[j], this->layers[this->layers.size() - 1][j].getLastX());
            d.push_back(this->layers[this->layers.size() - 1][j].getLastD());
        }

        if (this->layers.size() == 1)
            continue;

        // 은닉 레이어 학습
        for (std::size_t j = this->layers.size() - 2; j >= 0; j--)
        {
            std::vector<double> new_d;

            for (std::size_t k = 0; k < this->layers[j].size(); k++)
            {
                std::vector<double> linked_w;
                for (std::size_t n = 0; n < this->layers[j + 1].size(); n++)
                {
                    linked_w.push_back(this->layers[j + 1][n].getWeight()[k]);
                }

                if (linked_w.size() != d.size())
                {
                    std::cerr << "linked_w.size() != d.size()" << std::endl;
                }

                double e_hidden = 0.0;
                for (std::size_t n = 0; n < linked_w.size(); n++)
                {
                    e_hidden += linked_w[n] * d[n];
                }

                this->layers[j][k].train(a, e_hidden, this->layers[j][k].getLastX());
                new_d.push_back(this->layers[j][k].getLastD());
            }

            if (j == 0)
            {
                break;
            }

            d = new_d;
        }
    }
}