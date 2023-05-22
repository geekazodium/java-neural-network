package com.geekazodium.handdrawndigitstuff;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.LeakyRelU;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;

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
        input.setBounds(0,0,MIN_HEIGHT,MIN_HEIGHT);

        appWindow.addWindowListener(new AppWindowListener(this));

        JPanel panel = new JPanel();
        appWindow.add(panel);
        JLabel test = new JLabel("this is a");
        test.setFont((new Font("Serif", Font.PLAIN, 30)));
        panel.add(test);
        panel.setBounds(MIN_HEIGHT,MIN_HEIGHT/2-100,300,100);
        appWindow.setLayout(null);

        NeuralNetwork neuralNetwork;
        try {
            neuralNetwork = NeuralNetwork.deserialize(new File("Network-784-200-100-50-10.json"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        running = true;
        appWindow.paint(appWindow.getGraphics());
        appWindow.addKeyListener(input);
        appWindow.setVisible(true);
        int frameCounter = 0;
        while (running){
            input.paint(input.getGraphics());
            frameCounter++;
            if(frameCounter>=60){
                float[] in = new float[28*28];
                for (int i = 0; i < input.content.length; i++) {
                    in[i] = input.content[i]/256f;
                }
                float[] out = neuralNetwork.evaluate(in,new LeakyRelU());
                float highestVal = 0.1f;
                int highestIndex = -1;
                for (int number = 0; number < out.length; number++) {
                    if(out[number] > highestVal){
                        highestIndex = number;
                        highestVal = out[number];
                    }
                }
                test.setText("this is a "+highestIndex);
            }
            try {
                Thread.sleep(1000/60);
            } catch (InterruptedException e) {
                running = false;
            }
        }
        System.exit(0);
    }
}
