package com.traffic.model;

import com.traffic.config.TrainingConfig;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTMModelBuilder {

    public static MultiLayerNetwork build(int inputSize) {

        MultiLayerConfiguration config =
                new NeuralNetConfiguration.Builder()
                        .updater(new Adam(TrainingConfig.LEARNING_RATE))
                        .l2(TrainingConfig.L2)
                        .weightInit(WeightInit.XAVIER)
                        .list()

                        .layer(new LSTM.Builder()
                                .nIn(inputSize)
                                .nOut(128)
                                .activation(Activation.TANH)
                                //.dropOut(TrainingConfig.DROPOUT)
                                .build())

                        .layer(new LSTM.Builder()
                                .nOut(64)
                                .activation(Activation.TANH)
                                //.dropOut(TrainingConfig.DROPOUT)
                                .build())

                        .layer(new RnnOutputLayer.Builder()
                                .nOut(inputSize)
                                .activation(Activation.IDENTITY)
                                .lossFunction(LossFunctions.LossFunction.MSE)
                                .build())

                        .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        return model;
    }
}