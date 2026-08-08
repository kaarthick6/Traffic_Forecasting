package com.traffic.evaluation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Metrics {

    public static void print(INDArray pred, INDArray actual) {

        INDArray diff = pred.sub(actual);

        double mse = diff.mul(diff).meanNumber().doubleValue();
        double mae = Transforms.abs(diff).meanNumber().doubleValue();
        double rmse = Math.sqrt(mse);

        System.out.println("MSE  : " + mse);
        System.out.println("MAE  : " + mae);
        System.out.println("RMSE : " + rmse);
    }
}