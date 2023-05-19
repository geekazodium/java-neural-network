package com.geekazodium.handdrawndigitstuff;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;

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
