from data_loader  import data_load
import autokeras as ak

from sklearn.model_selection import KFold

inputs, targets = data_load()

# Define the K-fold Cross Validator
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(block_type=None,
        # Normalize the dataset.
        normalize=True,
        # Do not do data augmentation.
        augment=True,
    )(input_node)
    output_node = ak.ClassificationHead()(output_node)
    clf = ak.AutoModel(
        inputs=input_node, outputs=output_node, overwrite=True, max_trials=10
    )
    clf.fit(inputs[train], targets[train], epochs=10)

    # Predict with the best model.
    predicted_y = clf.predict(inputs[test])
    print(predicted_y)

    # Evaluate the best model with testing data.
    print(clf.evaluate(inputs[test], targets[test]))
            
    # Generate a print
    print('------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    
    # Increase fold number
    fold_no = fold_no + 1


    
