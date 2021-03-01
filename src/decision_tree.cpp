#include "decision_tree.h"

double calc_info_entropy(const target_type& targets) {
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
        const target_type& left_target,
        const target_type& right_target
) {

    double s1 = calc_info_entropy(left_target);
    double s2 = calc_info_entropy(right_target);

    int left_size =  left_target.size();
    int right_size = right_target.size();
    int node_size = left_size + right_size;

    return s0 - (s1*left_size + s2*right_size) / node_size;
}


TreeNode::TreeNode(
    dataset& ds,
    std::vector<int> objects,
    std::vector<bool> used_features,
    double entropy_threshold,
    int depth,
    int max_depth,
    random_gen_type& random_gen,
    bool use_random_features
)
    : ds_m(ds)
    , entropy_threshold_m(entropy_threshold)
    , objects_m(std::move(objects))
    , used_features_m(std::move(used_features))
    , depth_m(depth)
    , max_depth_m(max_depth)
    , random_gen_m(random_gen)
    , use_random_features_m(use_random_features)
{
    build();
}


TreeNode::TreeNode(
    dataset& ds,
    double entropy_threshold,
    int depth,
    int max_depth,
    random_gen_type& random_gen,
    bool use_random_features
)
    : ds_m(ds)
    , entropy_threshold_m(entropy_threshold)
    , depth_m(depth)
    , max_depth_m(max_depth)
    , random_gen_m(random_gen)
    , use_random_features_m(use_random_features)
{
    for (int i = 0; i < ds_m.X.size(); i++) {
        objects_m.push_back(i);
    }
    used_features_m = std::vector(ds_m.X[0].size(), false);

    build();
}

bool TreeNode::check_stop_criteria() {
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

std::vector<int> TreeNode::get_leaf_objects(int ftr, bool get_true) {
    std::vector<int> leaf_objects;
    std::copy_if(
            objects_m.begin(), objects_m.end(),
            std::back_inserter(leaf_objects),
            [this, ftr, get_true](int pos){ return get_true == ds_m.X[pos][ftr]; }
    );
    return leaf_objects;
}

target_type TreeNode::get_leaf_targets(std::vector<int>& leaf_objects) {
    target_type leaf_targets;
    std::transform(
            leaf_objects.begin(), leaf_objects.end(),
            std::back_inserter(leaf_targets),
            [this](size_t pos) { return ds_m.y[pos]; }
    );
    return leaf_targets;
}

void TreeNode::add_leaf(std::shared_ptr<TreeNode>& leaf, bool is_left) {
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
                depth_m+1, max_depth_m,
                random_gen_m,
                use_random_features_m
        );
    }
}

std::vector<bool> TreeNode::generate_feature_mask(int n_features, int n_leave) {
    std::vector<bool> feature_mask(n_features, false);
    int n_filled = 0;

    while ((n_features - n_filled) > n_leave) {
        std::uniform_int_distribution<> uid(0, n_features-1);
        int pos = uid(random_gen_m);

        feature_mask[pos] = true;
        n_filled = std::accumulate(feature_mask.begin(), feature_mask.end(), 0);
    }

    return feature_mask;
}

void TreeNode::build() {
    target_type node_targets = get_leaf_targets(objects_m);

    node_entropy = calc_info_entropy(node_targets);
    num_used_features_m = std::accumulate(used_features_m.begin(), used_features_m.end(), 0);

    if (check_stop_criteria()) {
        return;
    }

    std::map<int, double> info_gains;

    std::vector<int> node_features;
    int n_features = static_cast<int>(ds_m.X[0].size()) - num_used_features_m;
    int n_leave = static_cast<int>(sqrt(n_features));
    std::vector<bool> feature_mask = generate_feature_mask(n_features, n_leave);
    // TODO ugly code, rewrite
    int ftr_count = 0;
    for (int i = 0; i < ds_m.X[0].size(); ++i) {
        if (!used_features_m[i]) {
            if (feature_mask[ftr_count] || !use_random_features_m) {
                node_features.push_back(i);
            }
            ++ftr_count;
        }
    }

    for (auto& ftr: node_features) {
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

int TreeNode::get_answer() {
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


Tree::Tree(
        double entropy_threshold,
        int max_depth,
        bool use_random_features,
        int random_state
)
        : entropy_threshold_m(entropy_threshold)
        , max_depth_m(max_depth)
        , use_random_features_m(use_random_features)
        , random_state_m(random_state)
        , random_gen_m(random_gen_type(random_state_m))
{}

void Tree::fit(dataset& tr_ds) {
    root_m = std::make_shared<TreeNode>(
        tr_ds,
        entropy_threshold_m,
        1, max_depth_m,
        random_gen_m,
        use_random_features_m
    );
}

void Tree::fit(
        dataset& tr_ds,
        std::vector<int>&& objects,
        std::vector<bool>&& used_features
) {
    root_m = std::make_shared<TreeNode>(
        tr_ds,
        objects,
        used_features,
        entropy_threshold_m,
        1, max_depth_m,
        random_gen_m,
        use_random_features_m
    );
}


target_type Tree::predict(feature_matrix_type& X) const {
    target_type preds;

    for (auto& x: X) {
        preds.push_back(predict(x));
    }
    return preds;
}

int Tree::predict(feature_type& x) const {
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
