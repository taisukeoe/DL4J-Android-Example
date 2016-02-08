package org.deeplearning4j.androidexamples;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Path;

import com.example.android.common.logger.Log;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    protected Iterator<URI> labelIter;
    protected Configuration conf;
    protected File currentFile;
    public List<String> labels = new ArrayList<>();
    protected boolean appendLabel = false;
    protected Collection<Writable> record = new ArrayList<>();
    protected final List<String> allowedFormats = Arrays.asList("tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG");
    protected boolean hitImage = false;

    protected InputSplit inputSplit;
    protected Map<String, String> fileNameMap = new LinkedHashMap<>();
    protected String pattern; // Pattern to split and segment file name, pass in regex
    protected int patternPosition = 0;

    public final static String WIDTH = NAME_SPACE + ".width";
    public final static String HEIGHT = NAME_SPACE + ".height";
    public final static String CHANNELS = NAME_SPACE + ".channels";

    private static Logger log = LoggerFactory.getLogger(AndroidImageRecordReader.class);


    public AndroidImageRecordReader() {
    }

    public AndroidImageRecordReader(int width, int height, int channels, List<String> labels) {
        this(width, height, channels, false);
        this.labels = labels;
    }

    public AndroidImageRecordReader(int width, int height, int channels, boolean appendLabel) {
        this.appendLabel = appendLabel;
     }


    public AndroidImageRecordReader(int width, int height, int channels, boolean appendLabel, List<String> labels) {
        this(width, height, channels, appendLabel);
        this.labels = labels;
    }

    public AndroidImageRecordReader(int width, int height, int channels, boolean appendLabel, String pattern, int patternPosition) {
        this(width, height, channels, appendLabel);
        this.pattern = pattern;
        this.patternPosition = patternPosition;
    }

    public AndroidImageRecordReader(int width, int height, int channels, boolean appendLabel, List<String> labels, String pattern, int patternPosition) {
        this(width, height, channels, appendLabel, labels);
        this.pattern = pattern;
        this.patternPosition = patternPosition;
    }

    protected boolean containsFormat(String format) {
        for (String format2 : allowedFormats)
            if (format.endsWith("." + format2))
                return true;
        return false;
    }


    @Override
    public void initialize(InputSplit split) throws IOException {
        inputSplit = split;
        if (split instanceof MultipleInputStreamInputSplit) {
            MultipleInputStreamInputSplit split2 = (MultipleInputStreamInputSplit) split;
            InputStream[] iss = split2.getIs();
            URI[] locations = split2.locations();

            iter = Arrays.asList(iss).iterator();
            labelIter = Arrays.asList(locations).iterator();
            for (URI loc:locations) {
                if (appendLabel) {
                    String parent = getLabelFromURI(loc);
                    int label = labels.indexOf(parent);
                    if (label >= 0)
                        record.add(new DoubleWritable(labels.indexOf(parent)));
                    else
                        throw new IllegalStateException("Illegal label " + parent);
                }
            }
        }
    }

    public String getLabelFromURI(URI loc){
        String path = loc.getPath();
        String parent = path;
        //could have been a uri
        if (path.contains("/")) {
            parent = path.substring(path.lastIndexOf('/') + 1);
        }
        return parent;
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.appendLabel = conf.getBoolean(APPEND_LABEL, false);
        this.labels = new ArrayList<>(conf.getStringCollection(LABELS));
        this.conf = conf;
        initialize(split);
    }


    @Override
    public Collection<Writable> next() {
        if (iter != null) {
            Collection<Writable> ret = new ArrayList<>();
            InputStream is = iter.next();
            String label = getLabelFromURI(labelIter.next());
            try {
                Bitmap bmp = BitmapFactory.decodeStream(is);
                INDArray row = toINDArrayGrey(bmp);
//                INDArray row = toINDArrayBGR(bmp);
                ret = RecordConverter.toRecord(row);
                if (appendLabel)
                    ret.add(new DoubleWritable(labels.indexOf(label)));
                is.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
            return ret;
        } else if (record != null) {
            hitImage = true;
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    @Override
    public boolean hasNext() {
        if (iter != null) {
            return iter.hasNext();
        } else if (record != null) {
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
     *
     * @param path the path to get the label from
     * @return the label for the given path
     */
    public String getLabel(String path) {
        if (fileNameMap != null && fileNameMap.containsKey(path)) return fileNameMap.get(path);
        return (new File(path)).getParentFile().getName();
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        try {
            initialize(inputSplit);
        } catch (Exception e) {
            throw new RuntimeException("Error during LineRecordReader reset", e);
        }
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        Bitmap bmp = BitmapFactory.decodeStream(dataInputStream);
        INDArray row = toINDArrayBGR(bmp);
        Collection<Writable> ret = RecordConverter.toRecord(row);
        if (appendLabel) ret.add(new DoubleWritable(labels.indexOf(getLabel(uri.getPath()))));
        return ret;
    }


    protected INDArray toINDArrayGrey(Bitmap image) {
        int width = image.getWidth();
        int height = image.getHeight();

        int plane = width * height;

        int[] pix = new int[plane];

        image.getPixels(pix, 0, width, 0, 0, width, height);

        int[] shape = new int[]{height, width};

        return Nd4j.create(new ImageByteBuffer(pix), shape);
    }
    
    protected INDArray toINDArrayBGR(Bitmap image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int plane = width * height;
        int channel = 3;


        int[] rawPix = new int[plane];

        image.getPixels(rawPix, 0, width, 0, 0, width, height);

        int[] pix = new int[plane * channel];

        for(int i = 0; i < plane; i++){
            int pixel = rawPix[i];
            pix[i] = (pixel & 0x00FF0000) >> 24;
            pix[i + plane] = (pixel & 0x0000FF00) >> 16;
            pix[i + plane*2] = (pixel & 0x000000FF) >> 8;
        }

        int[] shape = new int[]{height, width, channel};

        log.info("shape:" + Arrays.toString(shape) + " bufSize:" + pix.length);
        INDArray ret = Nd4j.create(new ImageByteBuffer(pix), shape);
        return ret.permute(2, 0, 1);
    }
}
