package org.deeplearning4j.androidexamples;

import org.canova.api.split.InputSplit;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;

/**
 * Created by taisukeoe on 16/01/26.
 */
public class MultipleInputStreamInputSplit implements InputSplit {
    private InputStream[] is;
    private URI[] location;
    /**
     * Instantiate with the given
     * file as a uri
     * @param is the input stream to use
     * @param labels the labels for each sample
     */

    public MultipleInputStreamInputSplit(InputStream[] is,String[] labels) {
        this.is = is;
        this.location = new URI[labels.length];
        for(int i = 0;i < labels.length;i++){
            location[i] = URI.create(labels[i]);
        }
    }

    public MultipleInputStreamInputSplit(InputStream[] is) {
        this.is = is;
    }

    @Override
    public long length() {
        throw new UnsupportedOperationException();
    }

    @Override
    public URI[] locations() {
        return location;
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    public InputStream[] getIs() {
        return is;
    }

    public void setIs(InputStream[] is) {
        this.is = is;
    }

    @Override
    public double toDouble(){
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat(){
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt(){
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong(){
        throw new UnsupportedOperationException();
    }
}
