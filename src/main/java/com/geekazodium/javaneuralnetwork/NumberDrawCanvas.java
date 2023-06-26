package com.geekazodium.javaneuralnetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class NumberDrawCanvas extends JPanel implements MouseListener, MouseMotionListener, KeyListener {
    public static int PIXEL_SIZE = 14;
    public static int PIXELS = 28;
    public int[] content = new int[PIXELS*PIXELS];

    public NumberDrawCanvas(){
        super();
        this.addMouseListener(this);
        this.addMouseMotionListener(this);
    }
    boolean init = false;

    @Override
    public void paint(Graphics g) {
        Graphics2D graphic2d = (Graphics2D) g;
        if(!init){
            graphic2d.setColor(Color.BLACK);
            int margins = Main.MIN_HEIGHT - PIXEL_SIZE * PIXELS;
            margins/=4;
            graphic2d.fillRect(margins, margins, PIXEL_SIZE * PIXELS, PIXEL_SIZE * PIXELS);
            init = true;
        }
        for (int i = 0; i < PIXELS * PIXELS; i++) {
            int x = i%PIXELS+1;
            int y = i/PIXELS+1;
            graphic2d.setColor(new Color(getOr0(content,i)+getOr0(content,i)*0x100+getOr0(content,i)*0x10000));
            graphic2d.fillRect(x*PIXEL_SIZE,y*PIXEL_SIZE,PIXEL_SIZE,PIXEL_SIZE);
        }
    }
    public int getOr0(int[] content, int index){
        if(index>=content.length)return 0;
        if(index<0)return 0;
        return content[index];
    }

    public void setPixel(int x,int y,byte value){
        System.out.println(x+","+y);
        if(x<0||x>=PIXELS)return;
        if(y<0||y>=PIXELS)return;
        int index = x+y*PIXELS;
        this.content[index] = value;
    }
    public void increasePixel(int x,int y,int value){
        if(x<0||x>=PIXELS)return;
        if(y<0||y>=PIXELS)return;
        int index = x+y*PIXELS;
        int brightness = Math.min(Math.max(value + this.content[index],0),255);
        this.content[index] = brightness;
    }

    public double distanceTo(float x,float y,int drawX, int drawY){
        double dx = x-drawX-0.5;
        double dy = y-drawY-0.5;
        return Math.sqrt(dx*dx+dy*dy);
    }

    @Override
    public void mouseClicked(MouseEvent e) {
    }

    float lastX = 0;
    float lastY = 0;

    public void update(MouseEvent e){
        float brushX = e.getX();
        float brushY = e.getY();
        float dx = lastX-brushX;
        float dy = lastY-brushY;
        if(Math.sqrt(dx*dx+dy*dy)<PIXEL_SIZE/2){
            return;
        }
        lastX = brushX;
        lastY = brushY;
        brushX/=PIXEL_SIZE;
        brushY/=PIXEL_SIZE;
        brushX-=1;
        brushY-=1;
        for (int y = 0; y < 9; y++) {
            for (int x = 0; x < 9; x++) {
                int drawX = (int) brushX-4+x;
                int drawY = (int) brushY-4+y;
                increasePixel(drawX, drawY, (int) Math.max(0.6*(255-distanceTo(brushX,brushY,drawX,drawY)*128),0));
            }
        }
    }

    boolean held = false;

    @Override
    public void mousePressed(MouseEvent e) {
        this.held = true;
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        this.held = false;
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        update(e);
    }

    @Override
    public void mouseMoved(MouseEvent e) {

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
            this.content = new int[PIXELS*PIXELS];
        }
    }
}
