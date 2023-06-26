package com.geekazodium.javaneuralnetwork.neuralnetwork;

import com.google.gson.JsonObject;

public interface SerializableToJsonLayer{
    JsonObject serializeToJson();

    void deserializeFromJson(JsonObject object);
}
