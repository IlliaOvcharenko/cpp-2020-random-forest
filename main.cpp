#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>

#include <boost/algorithm/string.hpp>

typedef std::vector<std::vector<bool>> feature_type;
typedef std::vector<int> target_type;

struct dataset {
    feature_type X;
    target_type y;

    dataset(): X(0), y(0) {}
};

void read_dataset(const std::string& filename, dataset& ds) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "file not opened" << std::endl;
    }

    std::string line;

    getline (file,line);
    std::vector<std::string> columns(0);
    boost::algorithm::split(columns, line, boost::is_any_of(","));

//    target_type targets(0);
    while (getline (file,line)) {
        std::vector<std::string> values_str(0);
        std::vector<bool> values_float(0);
        boost::algorithm::split(values_str, line, boost::is_any_of(","));

        ds.y.push_back(std::stoi(values_str.back()));
        values_str.pop_back();

        for (const auto& val: values_str) {
            values_float.push_back(std::stoi(val));
        }
        ds.X.push_back(values_float);
    }
    file.close();
//    ds.y = targets;
}

double calc_info_entropy(const std::vector<int>& targets) {
    std::cout << (double)(targets.size()) << std::endl;

    double s = 0.0;

//    std::cout << "calc info entropy" << std::endl;
//    std::cout << targets.size() << std::endl;
    if (targets.empty()) return s;
//    for (auto& el: targets) {
//        std::cout << el << " ";
//    }
//    std::cout << std::endl;
    int n_classes = *std::max_element(targets.begin(), targets.end()) + 1;
//    std::cout << "calc info entropy 2" << std::endl;

    std::vector<int> count(n_classes, 0);
    for (const auto& t: targets) {
        ++count[t];
    }

    for (int cls = 0; cls < n_classes; ++cls) {
        std::cout << "cls: " << cls << " count: " << count[cls] << std::endl;
        int n_target = targets.size();
        std::cout << (double)(n_target) << std::endl;
        std::cout << static_cast<double>(count[cls]) / targets.size() << std::endl;
        double p = static_cast<double>(count[cls]) / targets.size();
        std::cout << "prob: " << p << " " << count[cls] << " " << targets.size() << std::endl;
        if (p <= 0.0) continue;
        s -= p * log2(p);
    }
    return s;
}

double calc_info_gain(
        double s0,
        target_type& left_target,
        target_type& right_target
    ) {
    std::cout << "calc info gain " << std::endl;

    double s1 = calc_info_entropy(left_target);
    double s2 = calc_info_entropy(right_target);

    int left_size =  left_target.size();
    int right_size = right_target.size();
    int node_size = left_size + right_size;

    return s0 - (s1*left_size + s2*right_size) / node_size;
}


class TreeNode {
public:
    std::shared_ptr<TreeNode> left_m;
    std::shared_ptr<TreeNode> right_m;
    float entropy_threshold_m = 0.0;
    std::vector<int> objects_m;
    dataset& ds_m;
    std::vector<bool> used_features_m;
    int ftr_to_split_m;


    explicit TreeNode(dataset& ds, std::vector<int> objects, std::vector<bool> used_features, float entropy_threshold)
        : ds_m(ds)
        , entropy_threshold_m(entropy_threshold)
        , objects_m(std::move(objects))
        , used_features_m(std::move(used_features))
        {
        build();
    }

    explicit TreeNode(dataset& ds, float entropy_threshold)
            : ds_m(ds)
            , entropy_threshold_m(entropy_threshold)
    {
        for (int i = 0; i < ds_m.X.size(); i++) {
            objects_m.push_back(i);
        }
        used_features_m = std::vector(ds_m.X[0].size(), false);

        build();
    }

    void build() {
        // TODO calculate initial entropy
        // TODO calculate information for every feature
        // TODO select feature with biggest IG
        // TODO create left and right children

        std::cout << "build node" << std::endl;
        target_type node_targets(objects_m.size(), 0);
        for (int i = 0; i < objects_m.size(); ++i) {
            node_targets[i] = ds_m.y[objects_m[i]];
        }
//        std::cout << objects_m.size() << std::endl;
//        std::cout << (double)(objects_m.size()) << std::endl;
        std::cout << ds_m.y.size() << std::endl;
        std::cout << static_cast<double>(ds_m.y.size()) << std::endl;
        std::cout << node_targets.size() << std::endl;
        std::cout << (double)(node_targets.size()) << std::endl;

//        for (auto& obj: objects_m) {
//            node_targets.push_back(ds_m.y[obj]);
//        }
//        std::transform(
//                objects_m.begin(), objects_m.end(),
//                std::back_inserter(node_targets),
//                [this](size_t pos) { return ds_m.y[pos]; }
//        );
        std::cout << "build node 1" << std::endl;

//        for (auto& t: node_targets) {
//            std::cout << t << " ";
//        }
//        std::cout << std::endl;

        double node_entropy = calc_info_entropy(node_targets);
        std::cout << "node entropy: " << node_entropy << std::endl;
        bool all_used = std::all_of(used_features_m.begin(), used_features_m.end(), [](bool v) { return v; });
        std::cout << "all used: " << all_used << std::endl;

        if (all_used || node_entropy < entropy_threshold_m) {
            return;
        }

        std::vector<double> info_gains(0);
        for (int ftr = 0; ftr < ds_m.X[0].size(); ++ftr) {
            if (used_features_m[ftr]) continue;

            std::vector<int> left_objects(0);
            std::copy_if(
                    objects_m.begin(), objects_m.end(),
                    std::back_inserter(left_objects),
                    [this, ftr](int pos){ return ds_m.X[pos][ftr]; }
            );
            target_type left_target(0);
            std::transform(
                    left_objects.begin(), left_objects.end(),
                    std::back_inserter(left_target),
                    [this](size_t pos) { return ds_m.y[pos]; }
            );
            std::cout << "there1" << std::endl;


            std::vector<int> right_objects(0);
            std::copy_if(
                    objects_m.begin(), objects_m.end(),
                    std::back_inserter(right_objects),
                    [this, ftr](int pos){ return !ds_m.X[pos][ftr]; }
            );
            target_type right_target(0);
            std::transform(
                    right_objects.begin(), right_objects.end(),
                    std::back_inserter(right_target),
                    [this](size_t pos) { return ds_m.y[pos]; }
            );
            std::cout << "there2" << std::endl;

            info_gains.push_back(calc_info_gain(
                    node_entropy,
                    left_target,
                    right_target
            ));

            std::cout << "there3" << std::endl;


        }

        std::cout << "information gain:" << std::endl;
        for (auto& ig: info_gains) {
            std::cout << ig << " ";
        }
        std::cout << std::endl;

        ftr_to_split_m = std::distance(
                info_gains.begin(),
                std::max_element(info_gains.begin(), info_gains.end())
        );
        std::cout << "feature to split: " << ftr_to_split_m << std::endl;

        std::vector<int> left_objects(0);
        std::copy_if(
                objects_m.begin(), objects_m.end(),
                std::back_inserter(left_objects),
                [this](int pos){ return ds_m.X[pos][ftr_to_split_m]; }
        );

        if (!left_objects.empty()) {
            std::cout << "build left node" << std::endl;
            std::vector<bool> left_used_features;
            std::copy(used_features_m.begin(), used_features_m.end(), std::back_inserter(left_used_features));
            left_used_features[ftr_to_split_m] = true;
            left_m = std::make_shared<TreeNode>(ds_m, left_objects, left_used_features, entropy_threshold_m);
        }

        std::vector<int> right_objects(0);
        std::copy_if(
                objects_m.begin(), objects_m.end(),
                std::back_inserter(right_objects),
                [this](int pos){ return !ds_m.X[pos][ftr_to_split_m]; }
        );

        if (!right_objects.empty()) {
            std::cout << "build right node" << std::endl;

            std::vector<bool> right_used_features;
            std::copy(used_features_m.begin(), used_features_m.end(), right_used_features.begin());
            right_used_features[ftr_to_split_m] = true;
            right_m = std::make_shared<TreeNode>(ds_m, right_objects, right_used_features, entropy_threshold_m);
        }
    }

    int get_answer() {
        // TODO select object and get mean target
        target_type node_targets;
        std::transform(
                objects_m.begin(), objects_m.end(),
                std::back_inserter(node_targets),
                [this](size_t pos) { return ds_m.y[pos]; }
        );

        int n_classes = *std::max_element(node_targets.begin(), node_targets.end()) + 1;
        std::vector<int> count(n_classes, 0);
        for (auto& t: node_targets) {
            ++count[t];
        }

        int answer = std::distance(
                count.begin(),
                std::max_element(count.begin(), count.end())
        );

        return answer;
    }
};


class Tree {
public:
    std::shared_ptr<TreeNode> root_m;
    explicit Tree() = default;

    void fit(dataset& tr_ds) {
        root_m = std::make_shared<TreeNode>(tr_ds, 0.5);
    }

    target_type predict(feature_type& X) {
        target_type preds;

        for (auto& x: X) {
            preds.push_back(predict(x));
        }
        return preds;
    }

    int predict(std::vector<bool>& x) {
        std::shared_ptr<TreeNode> node(root_m);

        while(true) {
            if (node->left_m == nullptr && node->right_m == nullptr) break;

            if (x[node->ftr_to_split_m]) {
                if (node->left_m == nullptr) break;
                node = node->left_m;
            } else {
                if (node->right_m == nullptr) break;
                node = node->right_m;
            }
        }

        return node->get_answer();
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


    Tree model{};
    model.fit(train_dataset);

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
