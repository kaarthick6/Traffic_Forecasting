package com.traffic.data;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Normalizer {

    public static INDArray minMax(INDArray data) {

        double min = data.minNumber().doubleValue();
        double max = data.maxNumber().doubleValue();

        return data.sub(min).div(max - min);
    }
}