package com.geekazodium.javaneuralnetwork;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class AppWindowListener extends WindowAdapter {

    private final Main appInstance;

    public AppWindowListener(Main appInstance){
        this.appInstance = appInstance;
    }

    @Override
    public void windowClosing(WindowEvent e) {
        appInstance.running = false;
    }
}
