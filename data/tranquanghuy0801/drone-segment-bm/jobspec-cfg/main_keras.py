from libs import training_keras
from libs import datasets
from libs import models_keras
from libs import inference_keras
from libs import scoring
from libs import unet
from libs import segunet

import wandb

if __name__ == '__main__':
    dataset = 'Deer_Photos_Labeled_New'  #  0.5 GB download
    #dataset = 'dataset-medium' # 9.0 GB download

    config = {
        'name' : 'combined-segnet-unet',
        'dataset' : dataset,
    }

    # wandb.init(config=config)

    # datasets.download_dataset(dataset)

    # train the model
    # model = models_keras.build_unet(size=320,encoder='vgg16')
    # model = unet.resnet50_unet(4,input_height=320,input_width = 320)
    model = models_keras.resnet50_segnet(4,input_height=320,input_width=320)
    # model = models_keras.vgg16_segnet(4,input_height=320,input_width=320)
    # model = segunet.segunet(input_shape=320,n_labels=4)
    print(model.summary())
    # model.load_weights("model-best.h5")
    # print("Load weights successfully")
    # training_keras.train_model(dataset, model)

    #  # use the train model to run inference on all test scenes
    # inference_keras.run_inference(dataset, model=model, basedir="predictions")

    # # scores all the test images compared to the ground truth labels then
    # # send the scores (f1, precision, recall) and prediction images to wandb
    # score, _ = scoring.score_predictions(dataset, basedir="predictions/resnet-segnet")
    # print(score)
    # wandb.log(score)

    # result = scoring.evaluate(model=model, dataset_path=dataset, num_classes=4)
    # print(result)
    # wandb.log(result)