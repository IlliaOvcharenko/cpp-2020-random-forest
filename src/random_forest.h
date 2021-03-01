#ifndef CPP_2020_RANDOM_FOREST_RANDOM_FOREST_H
#define CPP_2020_RANDOM_FOREST_RANDOM_FOREST_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <queue>
#include <thread>
#include <mutex>
#include <cmath>

#include "data.h"
#include "decision_tree.h"


class tree_queue {
public:
    std::queue<Tree> base_queue_m;
    std::mutex write_mutex_m;

    void push(Tree&& t);
    bool pop(Tree& t);
};

class RandomForest {
    std::vector<Tree> trees_m;
    int max_depth_m;
    int random_state_m;
    int n_estimators_m;

    double entropy_threshold_m = 0.0;

    int n_jobs_m;

public:
    RandomForest(int n_estimators, double entropy_threshold, int max_depth, int random_state, int n_jobs);

    void fit(dataset& tr_ds);

    void build_trees(dataset& tr_ds, tree_queue& todo_trees, tree_queue& out_trees);

    std::vector<int> generate_subset(int len, random_gen_type& random_gen);

    target_type predict(feature_matrix_type& X);

    int predict(feature_type& x);
};


#endif //CPP_2020_RANDOM_FOREST_RANDOM_FOREST_H
