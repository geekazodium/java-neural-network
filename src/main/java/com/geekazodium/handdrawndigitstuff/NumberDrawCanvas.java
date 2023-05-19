package com.geekazodium.handdrawndigitstuff;

import javax.swing.*;
import java.awt.*;

public class NumberDrawCanvas extends JPanel {
    public static int PIXEL_SIZE = 14;
    public static int PIXELS = 28;
    public int[] content = new int[]{0,0,255,1,0};
    @Override
    public void paint(Graphics g) {
        Graphics2D graphic2d = (Graphics2D) g;
        graphic2d.setColor(Color.BLACK);
        int margins = Main.MIN_HEIGHT- PIXEL_SIZE * PIXELS;
        margins/=4;
        graphic2d.fillRect(margins, margins, PIXEL_SIZE * PIXELS, PIXEL_SIZE * PIXELS);
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
}
