#include <iostream>
#include <string>
#include <cstdlib>
#include <queue>
#include <thread>
#include <mutex>
#include <cmath>

#include "src/data.h"
#include "src/decision_tree.h"
#include "src/metrics.h"


class tree_queue {
public:
    std::queue<Tree> base_queue_m;
    std::mutex write_mutex_m;

    void push(Tree&& t) {
        std::unique_lock<std::mutex> locker(write_mutex_m);
        base_queue_m.push(std::move(t));
    }

    bool pop(Tree& t) {
        std::unique_lock<std::mutex> locker(write_mutex_m);
        if (base_queue_m.empty()) { return false; }

        t = std::move(base_queue_m.front());
        base_queue_m.pop();
        return true;
    }
};

class RandomForest {
public:
    std::vector<Tree> trees_m;
    int max_depth_m;
    int random_state_m;
    int n_estimators_m;

    double entropy_threshold_m = 0.0;


    RandomForest(int n_estimators, int max_depth, int random_state, double entropy_threshold)
        : n_estimators_m(n_estimators)
        , max_depth_m(max_depth)
        , random_state_m(random_state)
        , entropy_threshold_m(entropy_threshold)
    {}

    void fit(dataset& tr_ds) {
        srand(random_state_m);

        tree_queue todo_trees;
        tree_queue out_trees;

        for (int i = 0; i < n_estimators_m; ++i) {
            todo_trees.base_queue_m.push(Tree(entropy_threshold_m, max_depth_m, true));
        }

        build_trees(std::ref(tr_ds), std::ref(todo_trees), std::ref(out_trees));

        while (!out_trees.base_queue_m.empty()) {
            trees_m.push_back(out_trees.base_queue_m.front());
            out_trees.base_queue_m.pop();
        }
    }

    void build_trees(dataset& tr_ds, tree_queue& todo_trees, tree_queue& out_trees) {
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

    std::vector<int> generate_subset(int len) {
        std::vector<int> subset;
        for (int i = 0; i < len; ++i) {
            subset.push_back(rand() % len);
        }
        return subset;
    }

    target_type predict(feature_matrix_type& X) {
        target_type preds;

        for (auto& x: X) {
            preds.push_back(predict(x));
        }
        return preds;
    }


    int predict(feature_type& x) {
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


};


int main() {
    std::string train_dataset_filename = "../data/train.csv";
    std::string val_dataset_filename = "../data/val.csv";
    dataset train_dataset;
    dataset val_dataset;

    read_dataset(train_dataset_filename, train_dataset);
    read_dataset(val_dataset_filename, val_dataset);

//    std::cout << train_dataset.X.size() << std::endl;
//    std::cout << train_dataset.X[0].size() << std::endl;
//    std::cout << train_dataset.y.size() << std::endl;


//    Tree model(0.1, 10, true);
//    model.fit(train_dataset);

    RandomForest model(20, 20, 42 ,0.1);
    model.fit(train_dataset);

    auto train_preds = model.predict(train_dataset.X);
    auto train_score = accuracy_score(train_dataset.y, train_preds);
    std::cout << "train score: " << train_score << std::endl;

    auto val_preds = model.predict(val_dataset.X);
    auto val_score = accuracy_score(val_dataset.y, val_preds);
    std::cout << "validation score: " << val_score << std::endl;

    return 0;
}
