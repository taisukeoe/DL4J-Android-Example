package org.deeplearning4j.androidexamples;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Path;

import org.apache.commons.io.FileUtils;
import org.canova.api.conf.Configuration;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.split.InputStreamInputSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.image.loader.ImageByteBuffer;
import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.Paths;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by taisukeoe on 16/01/19.
 */
public class AndroidImageRecordReader implements RecordReader {
    protected Iterator<InputStream> iter;
    protected Configuration conf;
    protected File currentFile;
    public List<String> labels  = new ArrayList<>();
    protected boolean appendLabel = false;
    protected Collection<Writable> record;
    protected final List<String> allowedFormats = Arrays.asList("tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG");
    protected boolean hitImage = false;
    protected ImageLoader imageLoader;
    protected InputSplit inputSplit;
    protected Map<String,String> fileNameMap = new LinkedHashMap<>();
    protected String pattern; // Pattern to split and segment file name, pass in regex
    protected int patternPosition = 0;

    public final static String WIDTH = NAME_SPACE + ".width";
    public final static String HEIGHT = NAME_SPACE + ".height";
    public final static String CHANNELS = NAME_SPACE + ".channels";
    
    public AndroidImageRecordReader() {
    }

    public AndroidImageRecordReader(int width, int height,int channels, List<String> labels) {
        this(width, height,channels,false);
        this.labels = labels;
    }

    public AndroidImageRecordReader(int width, int height,int channels, boolean appendLabel) {
        this.appendLabel = appendLabel;
        imageLoader = new ImageLoader(width,height,channels);
    }


    public AndroidImageRecordReader(int width, int height,int channels, boolean appendLabel, List<String> labels) {
        this(width,height,channels,appendLabel);
        this.labels = labels;
    }

    public AndroidImageRecordReader(int width, int height,int channels, boolean appendLabel, String pattern, int patternPosition) {
        this(width,height,channels,appendLabel);
        this.pattern = pattern;
        this.patternPosition = patternPosition;
    }

    public AndroidImageRecordReader(int width, int height,int channels, boolean appendLabel, List<String> labels, String pattern, int patternPosition) {
        this(width,height,channels,appendLabel, labels);
        this.pattern = pattern;
        this.patternPosition = patternPosition;
    }

    protected boolean containsFormat(String format) {
        for(String format2 : allowedFormats)
            if(format.endsWith("." + format2))
                return true;
        return false;
    }


    @Override
    public void initialize(InputSplit split) throws IOException{
        inputSplit = split;
        if(split instanceof InputStreamInputSplit) {
            InputStreamInputSplit split2 = (InputStreamInputSplit) split;
            InputStream is = split2.getIs();
            URI[] locations = split2.locations();
            INDArray load = imageLoader.asRowVector(is);
            record = RecordConverter.toRecord(load);
            for (int i = 0; i < load.length(); i++) {
                if (appendLabel) {
                    Path path = Paths.get(locations[0]);
                    String parent = path.getParent().toString();
                    //could have been a uri
                    if (parent.contains("/")) {
                        parent = parent.substring(parent.lastIndexOf('/') + 1);
                    }
                    int label = labels.indexOf(parent);
                    if (label >= 0)
                        record.add(new DoubleWritable(labels.indexOf(parent)));
                    else
                        throw new IllegalStateException("Illegal label " + parent);
                }
            }
            is.close();
        }
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.appendLabel = conf.getBoolean(APPEND_LABEL,false);
        this.labels = new ArrayList<>(conf.getStringCollection(LABELS));
        imageLoader = new ImageLoader(conf.getInt(WIDTH,28),conf.getInt(HEIGHT,28),conf.getInt(CHANNELS,1));
        this.conf = conf;
        initialize(split);
    }


    @Override
    public Collection<Writable> next() {
        if(iter != null) {
            Collection<Writable> ret = new ArrayList<>();
            InputStream is = iter.next();

            try {
                Bitmap bmp = BitmapFactory.decodeStream(is);
                INDArray row = toINDArrayBGR(bmp);
                ret = RecordConverter.toRecord(row);
                if(appendLabel)
                    ret.add(new DoubleWritable(labels.indexOf(image.getParentFile().getName())));
            } catch (Exception e) {
                e.printStackTrace();
            }
            return ret;
        }
        else if(record != null) {
            hitImage = true;
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    @Override
    public boolean hasNext() {
        if(iter != null) {
            return iter.hasNext();
        }
        else if(record != null) {
            return !hitImage;
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }


    /**
     * Get the label from the given path
     * @param path the path to get the label from
     * @return the label for the given path
     */
    public String getLabel(String path) {
        if(fileNameMap != null && fileNameMap.containsKey(path)) return fileNameMap.get(path);
        return (new File(path)).getParentFile().getName();
    }

    @Override
    public List<String> getLabels(){
        return labels; }

    @Override
    public void reset() {
        if(inputSplit == null) throw new UnsupportedOperationException("Cannot reset without first initializing");
        try{
            initialize(inputSplit);
        }catch(Exception e){
            throw new RuntimeException("Error during LineRecordReader reset",e);
        }
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream ) throws IOException {
        Bitmap bmp = BitmapFactory.decodeStream(dataInputStream);
        INDArray row = toINDArrayBGR(bmp);
        Collection<Writable> ret = RecordConverter.toRecord(row);
        if(appendLabel) ret.add(new DoubleWritable(labels.indexOf(getLabel(uri.getPath()))));
        return ret;
    }

    protected INDArray toINDArrayBGR(Bitmap image) {
        int width = image.getWidth();
        int height = image.getHeight();

        int[] pix = new int[width * height];

        image.getPixels(pix,0,width,0,0,width,height);

        int[] shape = new int[]{height, width, 3};
        INDArray ret = Nd4j.create(new ImageByteBuffer(pix), shape);
        return ret.permute(2, 0, 1);
    }
}
