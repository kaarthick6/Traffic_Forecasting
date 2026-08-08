package com.traffic.trainer;

import com.traffic.config.TrainingConfig;
import com.traffic.evaluation.Metrics;
import com.traffic.util.ModelSaver;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;

public class LSTMTrainer {

    public static void train(MultiLayerNetwork model,
                             DataSet trainData,
                             DataSet valData) throws Exception {

        double bestScore = Double.MAX_VALUE;
        int patienceCounter = 0;
        MultiLayerNetwork bestModel = model.clone();

        for (int epoch = 0; epoch < 200; epoch++) {

            model.fit(trainData);

            // Validation loss
            double score = model.score(valData);

            System.out.println("Epoch " + epoch +
                    " Validation Loss: " + score);

            if (score < bestScore) {
                bestScore = score;
                bestModel = model.clone();
                patienceCounter = 0;
            } else {
                patienceCounter++;
            }

            if (patienceCounter >= TrainingConfig.EARLY_STOPPING_PATIENCE) {
                System.out.println("Early stopping triggered.");
                break;
            }
        }

        INDArray predictions =
                bestModel.output(valData.getFeatures());

        Metrics.print(predictions, valData.getLabels());


        ModelSaver.save(bestModel, "lstm_20sensor_model.zip");

        System.out.println("Best Validation Score: " + bestScore);
    }
}