#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

typedef std::vector<std::vector<float>> feature_type;
typedef std::vector<float> target_type;

struct dataset {
    feature_type X;
    target_type y;
};

void read_dataset(const std::string& filename, dataset& ds) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "file not opened" << std::endl;
    }

    std::string line;

    getline (file,line);
    std::vector<std::string> columns;
    boost::algorithm::split(columns, line, boost::is_any_of(","));

    target_type targets;
    while (getline (file,line)) {
        std::vector<std::string> values_str;
        std::vector<float> values_float;
        boost::algorithm::split(values_str, line, boost::is_any_of(","));

        targets.push_back(std::stof(values_str.back()));
        values_str.pop_back();

        for (const auto& val: values_str) {
            values_float.push_back(std::stof(val));
        }
        ds.X.push_back(values_float);
    }
    file.close();
    ds.y = targets;
}

class TreeNode {
public:
    TreeNode& left;
    TreeNode& right;
};

class Tree {
public:
    TreeNode& root;

    explicit Tree(TreeNode &root) : root(root) { }

    void fit(dataset& tr_ds) {

    }

    target_type predict(feature_type& X) {
//        return ;
    }
};

//class BaggingTree {
//public:
//
//};



int main() {
    std::cout << "Hello, World!" << std::endl;
    std::string train_dataset_filename = "../data/train.csv";
    std::string val_dataset_filename = "../data/val.csv";
    dataset train_dataset;
    dataset val_dataset;

    read_dataset(train_dataset_filename, train_dataset);
    read_dataset(val_dataset_filename, val_dataset);

    std::cout << train_dataset.X.size() << std::endl;
    std::cout << train_dataset.X[0].size() << std::endl;
    std::cout << train_dataset.y.size() << std::endl;

//    for (auto& el: train_dataset.X[0]) {
//        std::cout << el << " ";
//    }
//    std::cout << std::endl;
//
//
//    for (auto& el: train_dataset.y) {
//        std::cout << el << " ";
//    }
//    std::cout << std::endl;
    return 0;
}
