#ifndef CPP_2020_RANDOM_FOREST_DECISION_TREE_H
#define CPP_2020_RANDOM_FOREST_DECISION_TREE_H

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

#include "data.h"


double calc_info_entropy(const target_type& targets);

double calc_info_gain(
        double s0,
        const target_type& left_target,
        const target_type& right_target
);


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
    );

    explicit TreeNode(dataset& ds, double entropy_threshold, int depth, int max_depth);

    bool check_stop_criteria();

    std::vector<int> get_leaf_objects(int ftr, bool get_true);

    target_type get_leaf_targets(std::vector<int>& leaf_objects);

    void add_leaf(std::shared_ptr<TreeNode>& leaf, bool is_left);

    void build();

    int get_answer();
};


class Tree {
    double entropy_threshold_m = 0.0;
    int max_depth_m;

public:
    std::shared_ptr<TreeNode> root_m;

    Tree(double entropy_threshold, int max_depth);

    void fit(dataset& tr_ds);

    target_type predict(feature_matrix_type& X) const;

    int predict(feature_type& x) const;
};

#endif //CPP_2020_RANDOM_FOREST_DECISION_TREE_H
