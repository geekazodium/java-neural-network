package com.geekazodium.javaneuralnetwork.utils;

public class ConsoleStylizer {
    public static String lineDivider(int length,String title){
        if(length<title.length()+2) return title;
        int titleLength = title.length();
        int repeat = (length - titleLength) / 2;
        return "=+".repeat(repeat/2)+title+"+=".repeat(repeat/2+repeat%2);
    }

    public static String formatByteSize(long bytes){
        String[] suffixes = new String[]{"B","B","KB","MB","GB","TB"};
        int c = 0;
        double b = bytes;
        double lastB = b;
        while (b > 1){
            lastB = b;
            b = b/1024d;
            c++;
        }
        return lastB+suffixes[c];
    }
}
