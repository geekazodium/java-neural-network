package com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes;

import java.util.*;

public class TrainingText {
    private final List<Integer> data;
    private final int chunkSize;
    private final Random random;
    private final int bound;
    public Map<Character,Integer> characterSet = new HashMap<>();
    public Map<Integer,Character> inverseCharset = new HashMap<>();

    public TrainingText(String s, String charSet, int chunkSize) {
        char[] charArray = s.toCharArray();
        int count = 0;
        for(char c: ((((char) 0))+charSet).toCharArray()){
            if(!characterSet.containsKey(c)) {
                characterSet.put(c, count);
                inverseCharset.put(count, c);
                count++;
            }
        }

        int dataLength = s.length();
        ArrayList<Integer> data = new ArrayList<>();
        for (int i = 0; i < dataLength; i++) {
            char key = charArray[i];
            if(!characterSet.containsKey(key)){
                System.out.println("INVALID CHARACTER: "+ key);
                continue;
            }
            data.add(characterSet.get(key));
        }
        this.data = new ArrayList<>(data);

        this.chunkSize = chunkSize;
        random = new Random();
        int size = this.data.size();
        bound = size - chunkSize;
    }

    public TextSection getExample(){
        int beginIndex = random.nextInt(0, bound);
        return new TextSection(data.subList(beginIndex,beginIndex+this.chunkSize),this.inverseCharset);
    }


    int index = 0;
    public TextSection getNextExample(){
        index += random.nextInt(this.chunkSize,this.chunkSize*2);
        index %= bound;
        return new TextSection(data.subList(index,index+this.chunkSize),this.inverseCharset);
    }

    public List<TextSection> getExamples(int batchSize) {
        TextSection[] ret = new TextSection[batchSize];
        for (int i = 0; i < batchSize; i++) {
            ret[i] = getExample();
        }
        return List.of(ret);
    }
}
