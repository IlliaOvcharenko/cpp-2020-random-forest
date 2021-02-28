#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>

#include <boost/algorithm/string.hpp>

typedef std::vector<bool> feature_type;
typedef std::vector<feature_type> feature_matrix_type;
typedef std::vector<int> target_type;

struct dataset {
    feature_matrix_type X;
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

//    target_type targets;
    while (getline (file,line)) {
        std::vector<std::string> values_str;
        std::vector<bool> values_bool;
        boost::algorithm::split(values_str, line, boost::is_any_of(","));

        ds.y.push_back(std::stoi(values_str.back()));
        values_str.pop_back();

        for (const auto& val: values_str) {
            values_bool.push_back(std::stoi(val));
        }
        ds.X.push_back(values_bool);
    }
    file.close();
}

double calc_info_entropy(const std::vector<int>& targets) {
    double s = 0.0;
    if (targets.empty()) return s;

    std::map<int, int> count;
    for (const auto& t: targets) {
        ++count[t];
    }

    for (auto& c: count) {
        double p = static_cast<double>(c.second) / targets.size();
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

    std::vector<int> objects_m;
    std::vector<bool> used_features_m;
    int num_used_features_m = 0;

    dataset& ds_m;
    int ftr_to_split_m{};

    int depth_m;
    int max_depth_m;

    double node_entropy = 0.0;
    double entropy_threshold_m = 0.0;


    explicit TreeNode(
            dataset& ds,
            std::vector<int> objects,
            std::vector<bool> used_features,
            double entropy_threshold,
            int depth,
            int max_depth
        )
        : ds_m(ds)
        , entropy_threshold_m(entropy_threshold)
        , objects_m(std::move(objects))
        , used_features_m(std::move(used_features))
        , depth_m(depth)
        , max_depth_m(max_depth)
        {
        build();
    }

    explicit TreeNode(dataset& ds, double entropy_threshold, int depth, int max_depth)
            : ds_m(ds)
            , entropy_threshold_m(entropy_threshold)
            , depth_m(depth)
            , max_depth_m(max_depth)
    {
        for (int i = 0; i < ds_m.X.size(); i++) {
            objects_m.push_back(i);
        }
        used_features_m = std::vector(ds_m.X[0].size(), false);

        build();
    }

    bool check_stop_criteria() {
        bool all_used = std::all_of(used_features_m.begin(), used_features_m.end(), [](bool v) { return v; });

        if (all_used) {
//            std::cout << "all feature used" << std::endl;
            return true;
        } else if (node_entropy < entropy_threshold_m) {
//            std::cout << "node entropy is small enough" << std::endl;
            return true;
        } else if (depth_m >= max_depth_m) {
//            std::cout << "max depth exceeded" << std::endl;
            return true;
        }

        return false;
    }

    std::vector<int> get_leaf_objects(int ftr, bool get_true) {
        std::vector<int> leaf_objects;
        std::copy_if(
            objects_m.begin(), objects_m.end(),
            std::back_inserter(leaf_objects),
            [this, ftr, get_true](int pos){ return get_true == ds_m.X[pos][ftr]; }
        );
        return leaf_objects;
    }

    target_type get_leaf_targets(std::vector<int>& leaf_objects) {
        target_type leaf_targets;
        std::transform(
            leaf_objects.begin(), leaf_objects.end(),
            std::back_inserter(leaf_targets),
            [this](size_t pos) { return ds_m.y[pos]; }
        );
        return leaf_targets;
    }

    void add_leaf(std::shared_ptr<TreeNode>& leaf, bool is_left) {
        std::vector<int> leaf_objects = get_leaf_objects(ftr_to_split_m, is_left);
        if (!leaf_objects.empty()) {
            std::vector<bool> leaf_used_features;
            std::copy(
                    used_features_m.begin(), used_features_m.end(),
                    std::back_inserter(leaf_used_features)
            );
            leaf_used_features[ftr_to_split_m] = true;

            leaf = std::make_shared<TreeNode>(
                    ds_m,
                    leaf_objects,
                    leaf_used_features,
                    entropy_threshold_m,
                    depth_m+1, max_depth_m
            );
        }
    }

    void build() {
        target_type node_targets = get_leaf_targets(objects_m);

        node_entropy = calc_info_entropy(node_targets);
        num_used_features_m = std::accumulate(used_features_m.begin(), used_features_m.end(), 0);

        if (check_stop_criteria()) {
            return;
        }

        std::map<int, double> info_gains;
        for (int ftr = 0; ftr < ds_m.X[0].size(); ++ftr) {
            if (used_features_m[ftr]) continue;

            std::vector<int> left_objects = get_leaf_objects(ftr, true);
            target_type left_target = get_leaf_targets(left_objects);

            std::vector<int> right_objects = get_leaf_objects(ftr, false);
            target_type right_target = get_leaf_targets(right_objects);

            info_gains[ftr] = calc_info_gain(
                node_entropy,
                left_target,
                right_target
            );
        }

        ftr_to_split_m = std::max_element(
            info_gains.begin(), info_gains.end(),
            [](auto& a, auto& b) { return a.second < b.second; }
        )->first;

        add_leaf(left_m, true);
        add_leaf(right_m, false);
    }

    int get_answer() {
        target_type node_targets;
            std::transform(
            objects_m.begin(), objects_m.end(),
            std::back_inserter(node_targets),
            [this](size_t pos) { return ds_m.y[pos]; }
        );

        std::map<int, int> answers;
        for (auto& t: node_targets) {
            ++answers[t];
        }

        int answer = std::max_element(
            answers.begin(), answers.end(),
            [](auto& a, auto& b) { return a.second < b.second; }
        )->first;

        return answer;
    }
};


class Tree {
public:
    std::shared_ptr<TreeNode> root_m;
    explicit Tree() = default;

    void fit(dataset& tr_ds) {
        root_m = std::make_shared<TreeNode>(tr_ds, 0.5, 1, 10);
    }

    target_type predict(feature_matrix_type& X) const {
        target_type preds;

        for (auto& x: X) {
            preds.push_back(predict(x));
        }
        return preds;
    }

    int predict(feature_type& x) const {
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


double accuracy_score(target_type& y_true, target_type& y_pred) {
    // TODO add assert in case of different sizes
    int n = y_true.size();
    std::vector<bool> is_equal(n, false);
    for (int i = 0; i < n; ++i) {
        is_equal[i] = (y_true[i] == y_pred[i]);
    }

    int tp = std::accumulate(is_equal.begin(), is_equal.end(), 0);
    return static_cast<double>(tp) / n;
}
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

//    for (auto& p: preds) {
//        std::cout << p << " ";
//    }
//    std::cout << std::endl;

    auto train_preds = model.predict(train_dataset.X);
    auto train_score = accuracy_score(train_dataset.y, train_preds);
    std::cout << "train score: " << train_score << std::endl;

    auto val_preds = model.predict(val_dataset.X);
    auto val_score = accuracy_score(val_dataset.y, val_preds);
    std::cout << "validation score: " << val_score << std::endl;

    return 0;
}
