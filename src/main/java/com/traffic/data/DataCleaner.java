package com.traffic.data;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class DataCleaner {

    public static int[] selectTopSensors(INDArray data, int topK) {

        int cols = data.columns();
        Map<Integer, Integer> zeroCount = new HashMap<>();

        for (int c = 0; c < cols; c++) {
            int zeros = 0;
            for (int r = 0; r < data.rows(); r++) {
                if (data.getDouble(r, c) == 0.0) zeros++;
            }
            zeroCount.put(c, zeros);
        }

        return zeroCount.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .limit(topK)
                .mapToInt(Map.Entry::getKey)
                .toArray();
    }
}