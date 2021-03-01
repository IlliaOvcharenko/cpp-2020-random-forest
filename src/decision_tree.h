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
#include <cstdlib>
#include <random>

#include "data.h"


double calc_info_entropy(const target_type& targets);

double calc_info_gain(
        double s0,
        const target_type& left_target,
        const target_type& right_target
);

typedef std::mt19937 random_gen_type;

class TreeNode {
    std::vector<int> objects_m;
    std::vector<bool> used_features_m;
    int num_used_features_m = 0;

    dataset& ds_m;

    int depth_m;
    int max_depth_m;

    double node_entropy = 0.0;
    double entropy_threshold_m = 0.0;

    bool use_random_features_m;
    random_gen_type& random_gen_m;

public:
    std::shared_ptr<TreeNode> left_m;
    std::shared_ptr<TreeNode> right_m;

    int ftr_to_split_m{};

    TreeNode(
        dataset& ds,
        std::vector<int> objects,
        std::vector<bool> used_features,
        double entropy_threshold,
        int depth,
        int max_depth,
        random_gen_type& random_gen,
        bool use_random_features=false
    );

    TreeNode(
        dataset& ds,
        double entropy_threshold,
        int depth,
        int max_depth,
        random_gen_type& random_gen,
        bool use_random_features=false
    );

    bool check_stop_criteria();

    std::vector<int> get_leaf_objects(int ftr, bool get_true);

    target_type get_leaf_targets(std::vector<int>& leaf_objects);

    void add_leaf(std::shared_ptr<TreeNode>& leaf, bool is_left);

    std::vector<bool> generate_feature_mask(int n_features, int n_leave);

    void build();

    int get_answer();
};


class Tree {
    double entropy_threshold_m = 0.0;
    int max_depth_m = 0.0;
    std::shared_ptr<TreeNode> root_m;
    bool use_random_features_m = false;
    random_gen_type random_gen_m;

public:
    int random_state_m;

    Tree() = default;

    Tree(double entropy_threshold, int max_depth, bool use_random_features=false, int random_state=42);

    void fit(dataset& tr_ds);

    void fit(
        dataset& tr_ds,
        std::vector<int>&& objects,
        std::vector<bool>&& used_features
    );

    target_type predict(feature_matrix_type& X) const;

    int predict(feature_type& x) const;
};

#endif //CPP_2020_RANDOM_FOREST_DECISION_TREE_H
