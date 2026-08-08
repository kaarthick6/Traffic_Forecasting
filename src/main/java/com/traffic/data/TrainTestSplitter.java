package com.traffic.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TrainTestSplitter {

    public static INDArray[] createSequences(
            INDArray data, int window) {

        int samples = data.rows() - window;
        int features = data.columns();

        // Input: [batch, features, timeSteps]
        INDArray X = Nd4j.zeros(samples, features, window);

        INDArray Y = Nd4j.zeros(samples, features, window);

        for (int i = 0; i < samples; i++) {

            for (int f = 0; f < features; f++) {


                for (int t = 0; t < window; t++) {
                    X.putScalar(
                            new int[]{i, f, t},
                            data.getDouble(i + t, f));
                }


                Y.putScalar(
                        new int[]{i, f, window - 1},
                        data.getDouble(i + window, f));
            }
        }

        return new INDArray[]{X, Y};
    }

    public static INDArray[] split(
            INDArray X,
            INDArray Y,
            double ratio) {

        int totalSamples = (int) X.size(0);
        int trainSize = (int)(totalSamples * ratio);

        INDArray trainX = X.get(
                NDArrayIndex.interval(0, trainSize),
                NDArrayIndex.all(),
                NDArrayIndex.all());

        INDArray trainY = Y.get(
                NDArrayIndex.interval(0, trainSize),
                NDArrayIndex.all(),
                NDArrayIndex.all());

        INDArray valX = X.get(
                NDArrayIndex.interval(trainSize, totalSamples),
                NDArrayIndex.all(),
                NDArrayIndex.all());

        INDArray valY = Y.get(
                NDArrayIndex.interval(trainSize, totalSamples),
                NDArrayIndex.all(),
                NDArrayIndex.all());

        return new INDArray[]{trainX, trainY, valX, valY};
    }
}