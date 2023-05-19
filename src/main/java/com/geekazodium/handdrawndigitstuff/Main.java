package com.geekazodium.handdrawndigitstuff;

import javax.swing.*;
import java.awt.*;

public class Main {
    public static void main(String[] args){
        if(args.length > 0) {
            if (args[0].equalsIgnoreCase("testing")) {
                testing();
                return;
            }
        }
        var instance = new Main();
        instance.start();
    }

    public static void testing(){

    }

    public boolean running = false;
    public static int MIN_WIDTH = 800;
    public static int MIN_HEIGHT = 450;

    public JFrame appWindow;
    public void start(){
        appWindow = new JFrame();
        appWindow.setBounds(0,0, MIN_WIDTH, MIN_HEIGHT);
        Dimension size = new Dimension(MIN_WIDTH, MIN_HEIGHT);
        appWindow.setMinimumSize(size);
        appWindow.setMaximumSize(size);
        appWindow.setName("hand drawn digit stuff");

        NumberDrawCanvas input = new NumberDrawCanvas();
        appWindow.add(input);

//        JLabel test = new JLabel("test");
//        appWindow.add(test);

        appWindow.addWindowListener(new AppWindowListener(this));
        running = true;
        appWindow.paint(appWindow.getGraphics());
        appWindow.setVisible(true);
        while (running){
            input.paint(input.getGraphics());
            try {
                Thread.sleep(1000/60);
            } catch (InterruptedException e) {
                running = false;
            }
        }
        System.exit(0);
    }
}
