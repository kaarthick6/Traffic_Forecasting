package com.traffic;

import com.traffic.config.TrainingConfig;
import com.traffic.data.*;
import com.traffic.model.*;
import com.traffic.trainer.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public class Main {

    public static void main(String[] args) throws Exception {

        // Load
        INDArray data = DataLoader.loadCSV("dataset/metr_la.csv");

        // Select best 20 sensors
        int[] bestSensors =
                DataCleaner.selectTopSensors(data,
                        TrainingConfig.TOP_SENSORS);

        INDArray reduced = data.getColumns(bestSensors);

        // Normalize
        reduced = Normalizer.minMax(reduced);

        // Create sequences
        INDArray[] sequences =
                TrainTestSplitter.createSequences(
                        reduced,
                        TrainingConfig.WINDOW_SIZE);

        INDArray features = sequences[0];
        INDArray labels = sequences[1];

        // Split
        INDArray[] split =
                TrainTestSplitter.split(
                        features,
                        labels,
                        TrainingConfig.TRAIN_RATIO);

        DataSet trainData =
                new DataSet(split[0], split[1]);

        DataSet valData =
                new DataSet(split[2], split[3]);

        // Build model
        var model =
                LSTMModelBuilder.build(
                        TrainingConfig.TOP_SENSORS);

        // Train
        LSTMTrainer.train(model, trainData, valData);
    }
}