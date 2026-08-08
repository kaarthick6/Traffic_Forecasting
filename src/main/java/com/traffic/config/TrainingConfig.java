package com.traffic.config;

public class TrainingConfig {

    public static final int WINDOW_SIZE = 12;
    public static final int TOP_SENSORS = 20;

    public static final double TRAIN_RATIO = 0.7;

    public static final double LEARNING_RATE = 0.001;
    public static final double L2 = 1e-6; //was 1e-4
    public static final double DROPOUT = 0.2;

    public static final int EARLY_STOPPING_PATIENCE = 10;
}