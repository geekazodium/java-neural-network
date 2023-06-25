package com.geekazodium.handdrawndigitstuff.neuralnetwork.trainingdatatypes;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TextSection {

    public static int inputLength;
    public final List<Integer> section;
    public final Map<Integer, Character> inverseCharset;

    public TextSection(List<Integer> section, Map<Integer, Character> charset) {
        this.inverseCharset = charset;
        this.section = section;
    }

    public static void setInputLength(int inputLength){
        TextSection.inputLength = inputLength;
    }

    public void log(){
        StringBuilder builder = new StringBuilder();
        for (Integer integer : this.section) {
            builder.append(inverseCharset.get(integer));
        }
        System.out.println(builder);
    }

    public float[] getData(int endIndex) {
        return chunkData(endIndex,section,inverseCharset.size(), section.size()-1);
    }

    public static float[] chunkData(int endIndex, List<Integer> section, int charsetSize, int sectionSize){

        float[] data = new float[sectionSize * charsetSize];

        int shiftLength = sectionSize - endIndex - 1;
        for (int i = 0; i < sectionSize; i++) {
            int offset = i * charsetSize;
            int listIndex = i - shiftLength;
            Integer characterToken;
            if(listIndex >= 0){
                characterToken = section.get(listIndex);
            }else {
                characterToken = 0;
            }
            if(characterToken == -1){
                characterToken = 0;
            }
            data[offset + characterToken] = 1.0f;
        }
//
//        StringBuilder debug = new StringBuilder();
//        int index = (sectionSize-endIndex-1)*charsetSize;
//        for (int i = 0; i < sectionSize; i++) {
//            if(i>endIndex)break;
//            Integer integer = section.get(i);
//            if(integer<0)continue;
//            debug.append(integer+",");
//            data[index + integer] = 1f;
//            index += charsetSize;
//        }
        return data;
    }

    public static float[] chunkString(String s, Map<Character, Integer> charset, Map<Integer, Character> inverseCharset){
        ArrayList<Integer> tokenized = new ArrayList<>();
        int stringLength = s.length();
        int i = stringLength - inputLength;
        for (int index = 0; index < inputLength; index++) {
            if(i+index<0){
                tokenized.add(-1);
                continue;
            }
            tokenized.add(charset.get(s.charAt(i+index)));
        }
        final ArrayList<Integer> section1 = new ArrayList<>(tokenized);
        return chunkData(Math.min(inputLength-1,stringLength), section1,inverseCharset.size(), section1.size());
    }
}
