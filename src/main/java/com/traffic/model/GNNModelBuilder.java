package com.traffic.model;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class GNNModelBuilder {

    public static MultiLayerNetwork build(int inputSize) {

        MultiLayerConfiguration config =
                new NeuralNetConfiguration.Builder()
                        .updater(new Adam(0.001))
                        .list()
                        .layer(new DenseLayer.Builder()
                                .nIn(inputSize)
                                .nOut(128)
                                .activation(Activation.RELU)
                                .build())
                        .layer(new OutputLayer.Builder()
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