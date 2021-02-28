#include <iostream>
#include <string>

#include "src/data.h"
#include "src/decision_tree.h"
#include "src/metrics.h"


int main() {
    std::string train_dataset_filename = "../data/train.csv";
    std::string val_dataset_filename = "../data/val.csv";
    dataset train_dataset;
    dataset val_dataset;

    read_dataset(train_dataset_filename, train_dataset);
    read_dataset(val_dataset_filename, val_dataset);

    std::cout << train_dataset.X.size() << std::endl;
    std::cout << train_dataset.X[0].size() << std::endl;
    std::cout << train_dataset.y.size() << std::endl;


    Tree model(0.5, 4);
    model.fit(train_dataset);

    auto train_preds = model.predict(train_dataset.X);
    auto train_score = accuracy_score(train_dataset.y, train_preds);
    std::cout << "train score: " << train_score << std::endl;

    auto val_preds = model.predict(val_dataset.X);
    auto val_score = accuracy_score(val_dataset.y, val_preds);
    std::cout << "validation score: " << val_score << std::endl;

    return 0;
}
