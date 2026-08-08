package com.traffic.util;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;

public class ModelSaver {

    public static void save(MultiLayerNetwork model, String name)
            throws Exception {

        ModelSerializer.writeModel(model, new File(name), true);
    }
}