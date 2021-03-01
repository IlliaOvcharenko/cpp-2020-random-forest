#include "random_forest.h"


void tree_queue::push(Tree&& t) {
    std::unique_lock<std::mutex> locker(write_mutex_m);
    base_queue_m.push(std::move(t));
}

bool tree_queue::pop(Tree& t) {
    std::unique_lock<std::mutex> locker(write_mutex_m);
    if (base_queue_m.empty()) { return false; }

    t = std::move(base_queue_m.front());
    base_queue_m.pop();
    return true;
}


RandomForest::RandomForest(int n_estimators, double entropy_threshold, int max_depth, int random_state, int n_jobs)
    : n_estimators_m(n_estimators)
    , max_depth_m(max_depth)
    , random_state_m(random_state)
    , entropy_threshold_m(entropy_threshold)
    , n_jobs_m(n_jobs)
{}

void RandomForest::fit(dataset& tr_ds) {
    tree_queue todo_trees;
    tree_queue out_trees;

    for (int i = 0; i < n_estimators_m; ++i) {
        todo_trees.base_queue_m.push(Tree(entropy_threshold_m, max_depth_m, true));
    }

    std::vector<std::thread> build_tree_threads;
    for (int j = 0; j < n_jobs_m; ++j) {
        build_tree_threads.emplace_back(
                &RandomForest::build_trees,
                this,
                std::ref(tr_ds),
                std::ref(todo_trees),
                std::ref(out_trees),
                random_state_m + j
        );
    }

    for (auto& t: build_tree_threads) {
        t.join();
    }

    while (!out_trees.base_queue_m.empty()) {
        trees_m.push_back(out_trees.base_queue_m.front());
        out_trees.base_queue_m.pop();
    }
}

void RandomForest::build_trees(dataset& tr_ds, tree_queue& todo_trees, tree_queue& out_trees, int seed) {
//    std::cout << seed << std::endl;
    srand(seed);

    while(true) {
        std::vector<int> objects_subset = generate_subset(tr_ds.X.size());

        int n_features = tr_ds.X[0].size();
        std::vector<bool> feature_mask(n_features, false);

        Tree tree;
        if(!todo_trees.pop(tree)) { break; }
        tree.fit(tr_ds, std::move(objects_subset), std::move(feature_mask));
        out_trees.push(std::move(tree));
    }
}

std::vector<int> RandomForest::generate_subset(int len) {
    std::vector<int> subset;
    for (int i = 0; i < len; ++i) {
        subset.push_back(rand() % len);
    }
    return subset;
}

target_type RandomForest::predict(feature_matrix_type& X) {
    target_type preds;

    for (auto& x: X) {
        preds.push_back(predict(x));
    }
    return preds;
}


int RandomForest::predict(feature_type& x) {
    std::map<int, int> answers;
    for (auto& t: trees_m) {
        ++answers[t.predict(x)];
    }

    int answer = std::max_element(
            answers.begin(), answers.end(),
            [](auto& a, auto& b) { return a.second < b.second; }
    )->first;

    return answer;
}
