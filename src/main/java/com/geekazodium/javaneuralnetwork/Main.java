package com.geekazodium.javaneuralnetwork;

import com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions.LeakyRelU;
import com.geekazodium.javaneuralnetwork.neuralnetwork.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import static com.geekazodium.javaneuralnetwork.NumberDrawCanvas.PIXELS;

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

//        JTextField textField = new CorrectionTextField(input);
//        JButton button = new CorrectionButton(input,textField);
//        panel.add(button);
//        appWindow.add(textField);

        NeuralNetwork neuralNetwork;
        try {
            neuralNetwork = NeuralNetwork.deserialize(new File("Deep_network_add.json"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        neuralNetwork.setActivationFunction(new LeakyRelU());

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
                float[] out = neuralNetwork.evaluate(in);
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

    private static void writeImageToFile(byte[] imageBytes, File file) {
        FileOutputStream outputStream;
        try {
            outputStream= new FileOutputStream(file);
        } catch (FileNotFoundException ex) {
            throw new RuntimeException(ex);
        }
        try {
            outputStream.write(imageBytes);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
        try {
            outputStream.close();
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    private static String generateHashString(byte[] imageContent, MessageDigest messageDigest) {
        StringBuilder hash = new StringBuilder();
        String lookup = "0123456789abcdef";
        byte[] digest = messageDigest.digest(imageContent);
        for (byte b : digest) {
            hash.append(lookup.charAt(b&0xf));
            hash.append(lookup.charAt((b>>4)&0xf));
        }
        return hash.toString();
    }

    private static byte[] writeImageContent(byte[] imageBytes, byte label) {
        byte[] content = new byte[imageBytes.length+1];
        System.arraycopy(imageBytes,0,content,1, imageBytes.length);
        content[0] = label;
        return content;
    }

    private static class CorrectionButton extends JButton implements MouseListener {
        private final NumberDrawCanvas input;
        private final JTextField textField;

        public CorrectionButton(NumberDrawCanvas input, JTextField textField) {
            this.input = input;
            this.addMouseListener(this);
            this.setText("output is incorrect");
            this.setFocusable(false);
            this.textField = textField;
        }

        @Override
        public void mouseClicked(MouseEvent e) {

            byte correctionNumber = (byte) Integer.parseInt(this.textField.getText());

            byte[] imageBytes = new byte[28*28];

            for (int i = 0; i < input.content.length; i++) {
                imageBytes[i] = ((byte) input.content[i]);
            }

            imageBytes = writeImageContent(imageBytes, correctionNumber);

            MessageDigest messageDigest;
            try {
                messageDigest = MessageDigest.getInstance("SHA-256");
            } catch (NoSuchAlgorithmException ex) {
                throw new RuntimeException(ex);
            }


            String fileName = "dataset" + File.separator + "inputFail-" + generateHashString(imageBytes, messageDigest);

            File file = new File(fileName);
            if(!file.exists()){
                try {
                    file.createNewFile();
                } catch (IOException ex) {
                    throw new RuntimeException(ex);
                }
            }

            writeImageToFile(imageBytes, file);
        }

        @Override
        public void mousePressed(MouseEvent e) {

        }

        @Override
        public void mouseReleased(MouseEvent e) {

        }

        @Override
        public void mouseEntered(MouseEvent e) {

        }

        @Override
        public void mouseExited(MouseEvent e) {

        }
    }

    private static class CorrectionTextField extends JTextField implements KeyListener {
        private final NumberDrawCanvas input;

        public CorrectionTextField(NumberDrawCanvas input){
            this.input = input;
            this.setBounds(550,100,100,30);
            this.addKeyListener(this);
        }


        @Override
        public void keyTyped(KeyEvent e) {

        }

        @Override
        public void keyPressed(KeyEvent e) {

        }

        @Override
        public void keyReleased(KeyEvent e) {
            if(e.getKeyCode() == KeyEvent.VK_SPACE) {
                input.content = new int[PIXELS* PIXELS];
                this.setText("");
            }
        }
    }

}
