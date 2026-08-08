package com.traffic.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

public class DataLoader {

    public static INDArray loadCSV(String path) throws Exception {

        List<double[]> rows = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;

        boolean firstLine = true;

        while ((line = br.readLine()) != null) {

            // Skip header
            if (firstLine) {
                firstLine = false;
                continue;
            }

            String[] tokens = line.split(",");

            // Skip timestamp column
            double[] values = new double[tokens.length - 1];

            for (int i = 1; i < tokens.length; i++) {

                String value = tokens[i].trim();

                if (value.isEmpty()) {
                    values[i - 1] = 0.0;   // handle empty cells
                } else {
                    values[i - 1] = Double.parseDouble(value);
                }
            }

            rows.add(values);
        }

        br.close();

        double[][] data = rows.toArray(new double[0][]);

        return Nd4j.create(data);
    }
}