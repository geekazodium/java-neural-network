package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.google.gson.JsonObject;

public interface SerializableToJsonLayer{
    JsonObject serializeToJson();

    void deserializeFromJson(JsonObject object);
}
